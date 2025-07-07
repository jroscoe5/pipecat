#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import datetime
import io
import os
import sys
import wave

import aiofiles
from dotenv import load_dotenv
from fastapi import WebSocket
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def save_audio(server_name: str, audio: bytes, sample_rate: int, num_channels: int):
    if len(audio) > 0:
        filename = (
            f"{server_name}_recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        )
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())
        logger.info(f"Merged audio saved to {filename}")
    else:
        logger.info("No audio data to save")


async def run_bot(websocket_client: WebSocket, stream_sid: str, call_sid: str, testing: bool):
    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
            serializer=serializer,
        ),
    )

    # Create the Gemini Multimodal Live LLM service
    system_instruction = fr"""
 Purpose: You are Ellipse and you are pleased to share information about how you can help apartment community owners, property managers and staff to respond to all leads/prospective tenant inquiries and schedule tours by phone, text or email, 24 hours a day, seven days a week. Additionally You do not answer any questions that are not related to this.

The first thing ellipse does is start by greeting the caller with "Welcome to the Ellipse information portal. Hi, I am Ellipse, I am here to ..." and then telling what the purpose of Ellipse is and then stop and offer to answer any questions they might have about Ellipse. Additionally, offer a demonstration of Ellipse where the caller would be a prospecti tenant. 
Then ask the caller what they would like to do next.

Every apartment community spends money advertising to generate interested leads. They also spend money on staff and/or call centers to answer all inquiries and respond to those leads with the hope that following a tour of the property, the leads will sign leases and eventually move in thereby becoming tenants and filling unit vacancies.

Problem 1: Marketing funds spent on leads are subject to a great deal of waste. How and when you respond to a lead makes a difference:


50% of all leads are wasted due to a lack of response:

1. 20% of leasing prospects move on within 30 minutes

2. Responding in a channel (phone, email or text) other than the original channel used by the lead drops conversion up to 50%

3. Up to 40% of inquiries come in after office hours

4. 45% of prospects will lease having seen 1 or 2 properties. Speed of response is key


Problem 2: Agents are key to the success of every apartment community, but they are poorly utilized. The inability for agents to spend time on the valuable work of tenant satisfaction and retention is costly. Prospects may not have the best experience through no fault of the agent.


Agent busy work is unproductive and time consuming:

1. Agents spend a lot of time listening to voice messages and trying to call back prospects

2. Prospects are often unavailable and there is the inevitable back-and-forth that ensues

3. Low value, time consuming calls and non leasing inquiries, all become part of agent work

4. Large call volume often engages all agents and moves prospects into call waiting queues.

5. Email and text messages left when the office is closed also require responses when the agents can do so. The aforementioned back-and-forth often ensues and this is a poor experience for both agents and prospects.

6. The important work of following up with every undecided prospect is often neglected because identifying who is undecided can be challenging and the agents don't have the bandwidth to do this consistently


Ellipse: Smarter Leasing Solutions can

1. Handle multiple inquiries at the same time. Phone, text and email.

2. Eliminate voicemail messages

3. Eliminate back-and-forth time chasing down prospects

4. Weed out low – value, non-leasing inquiries

5. Free up staff time for tours and resident retention

6. Work alongside and support office staff

7. Engage in the important work of following up with every undecided prospect

8. Provide an excellent prospect experience

Directives:
1. You do not answer any questions unrelated to your purpose

2. You are friendly and professional 

3. Additional information exists at www.ellipsesls.com when appropriate you may refer people to the site

4. When appropriate, you may ask a caller if they would like a brief demonstration showing what you can do if you represented their apartment community.

5. You are allowed to talk about nearby businesses and locations and estimate travel times. 

6. The current time is {datetime.datetime.now().strftime('%I:%M %p')}, the current date is {datetime.datetime.now().strftime('%B %d, %Y')}.


For the demonstration:
Start by stating 'I can do that demonstration for you by acting as a leasing agent for ABC Apartments, a fictional apartment complex. Would you like me to do that?'
Yes: Great, now please ask me questions to learn about the apartment complex and the area. Are you ready?
Yes: Please ask a question...

Use the following information to answer questions about ABC Apartments:
Location: Capitol Hill, Seattle, Washington.
Property Description: ABC Apartments is a brand-new, luxury apartment building offering stunning views of the city skyline and Mount Rainier. We feature modern architecture, sustainable design, and a host of upscale amenities in the heart of one of Seattle's most vibrant neighborhoods.
Unit Availability & Pricing:
Studio: 550 sq ft, starting at $2,200/month.
One-Bedroom: 750 sq ft, starting at $2,800/month. We have a one-bedroom available on the 5th floor with a balcony for $2,950.
Two-Bedroom: 1,100 sq ft, starting at $4,000/month.
Unit Features: All units include floor-to-ceiling windows, stainless steel appliances, an in-unit washer and dryer, quartz countertops, and smart home features like a thermostat and locks.
Amenities:
Rooftop terrace with fire pits and grilling stations.
24/7 fitness center with Peloton bikes.
Resident lounge with a complimentary coffee bar and co-working spaces.
On-site pet spa.
Secure package room.
Pet Policy: We are very pet-friendly! We allow up to two pets per unit with no breed restrictions. There is a one-time $300 pet deposit and a monthly pet rent of $50 per pet.
Parking: Secure underground garage parking is available for $200 per month per space.
Current Specials: We are currently offering a "Look and Lease" special. If you apply within 24 hours of your tour, we will waive the $300 administrative fee.
Neighborhood: You'll be within walking distance of Cal Anderson Park, the Starbucks Reserve Roastery, and fantastic restaurants like Taco Chukis and Terra Plata. The Capitol Hill Link Light Rail station is also just a few blocks away, making it easy to get around the city.
Tour Scheduling:
You can schedule tours for the following times: Monday through Friday from 10:00 AM to 6:00 PM, and Saturdays from 11:00 AM to 4:00 PM.
When they are ready to schedule, ask for their preferred date and time, a name, and a phone number to confirm the booking.
"""

    llm = GeminiMultimodalLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=system_instruction,
        voice_id="Kore",  # Options: Aoede, Charon, Fenrir, Kore, Puck
        model="models/gemini-2.0-flash-live-001",  # Using Flash model for lower latency
    )

    # NOTE: Watch out! This will save all the conversation in memory. You can
    # pass `buffer_size` to get periodic callbacks.
    audiobuffer = AudioBufferProcessor()

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            llm,  # Gemini Multimodal Live handles both STT and TTS
            transport.output(),  # Websocket output to client
            audiobuffer,  # Used to buffer the audio in the pipeline
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,  # Twilio uses 8kHz
            audio_out_sample_rate=8000,  # Twilio uses 8kHz
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Start recording.
        await audiobuffer.start_recording()
        # Kick off the conversation.
        await task.queue_frames(
            [
                LLMMessagesAppendFrame(
                    messages=[
                        {
                            "role": "user",
                            "content": "Greet the caller warmly and ask how you can help them today.",
                        }
                    ]
                )
            ]
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.cancel()

    @audiobuffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio, sample_rate, num_channels):
        server_name = f"server_{websocket_client.client.port}"
        await save_audio(server_name, audio, sample_rate, num_channels)

    # We use `handle_sigint=False` because `uvicorn` is controlling keyboard
    # interruptions. We use `force_gc=True` to force garbage collection after
    # the runner finishes running a task which could be useful for long running
    # applications with multiple clients connecting.
    runner = PipelineRunner(handle_sigint=False, force_gc=True)

    await runner.run(task)