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
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams
from pipecat.frames.frames import EndFrame, TTSSpeakFrame

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
            vad_analyzer=SileroVADAnalyzer(),
            serializer=serializer,
        ),
    )

    # Define the hang up function schema
    hang_up_function = FunctionSchema(
        name="hang_up_call",
        description="End the phone call when the user requests to hang up or says goodbye",
        properties={},
        required=[],
    )
    tools = ToolsSchema(standard_tools=[hang_up_function])

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"), 
        params=OpenAILLMService.InputParams(temperature=0.7),
        tools=tools
    )

    # Define the hang up function
    async def hang_up_call(params: FunctionCallParams):
        logger.info("User requested to hang up the call")
        # First acknowledge the request
        await tts.queue_frame(TTSSpeakFrame("Goodbye! Thank you for calling."))
        # Return result to complete the function call
        await params.result_callback({"status": "ending_call"})
        # Push EndFrame to trigger Twilio auto-hangup
        await task.queue_frames([EndFrame()])

    # Register the function with the LLM
    llm.register_function("hang_up_call", hang_up_call)
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"), audio_passthrough=True)
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="6f84f4b8-58a2-430c-8c79-688dad597532", #"bf0a246a-8642-498a-9950-80c35e9276b5",  #"008fa54c-4d6c-4cde-85a1-d450fe476085",#"bf0a246a-8642-498a-9950-80c35e9276b5",  # British Reading Lady
        push_silence_after_stop=testing,
        params=CartesiaTTSService.InputParams(
            speed='slow',
        )
    )

    import datetime
    system_instruction = """
<<Core Identity>>
You are Ellipse, an AI assistant that helps apartment communities respond to prospective tenant inquiries 24/7 via phone, text, or email. You schedule tours and answer questions about properties.
</</Core Identity>>

<<Conversation Guidelines>>
<Opening Interaction>
Start with: "Welcome to the Ellipse information portal. Hi, I'm Ellipse."
If interrupted, continue naturally from where you left off without restarting
Keep your initial explanation brief: "I help apartment communities connect with prospective tenants and schedule tours anytime, day or night."
Ask: "Would you like to learn more about how Ellipse works, or would you prefer to see a demonstration?"
</Opening Interaction>

<Natural Conversation Rules>
Be conversational, not robotic
Use natural language, not lists or bullet points
Vary your responses to avoid sounding scripted
Keep responses concise and focused on what was asked
Handle interruptions gracefully
Never restart your introduction if cut off
Answer specifically
Address only what was asked
Avoid information dumps
Use examples rather than listing features
Maintain professionalism
Stay friendly and helpful
Redirect off-topic questions politely
Never use special characters or formatting symbols
Stay on topic about Ellipse and apartment leasing
No special characters or formatting in responses
Keep the current date and time handy
Be helpful but redirect unrelated questions politely
Sound natural, not like you're reading from a script
</Natural Conversation Rules>

<Call Control>
You have the ability to end the phone call when requested. If the user:
- Explicitly asks to hang up or end the call
- Says goodbye in a way that indicates they want to end the conversation
- Says they're done or finished talking
Then use the hang_up_call function to end the call gracefully.
</Call Control>
</</Conversation Guidelines>>

<<Information for Responses>>
<What Ellipse Does>
Handle multiple inquiries at the same time. Phone, text and email.
Eliminate voicemail messages
Eliminate back-and-forth time chasing down prospects
Weed out low value, non-leasing inquiries
Free up staff time for tours and resident retention
Work alongside and support office staff
Engage in the important work of following up with every undecided prospect
Provide an excellent prospect experience
</What Ellipse Does>

<Why Ellipse is Needed>
Properties typically lose half their leads due to slow response times
Many inquiries come after hours when offices are closed
Ellipse ensures every lead gets immediate attention
Frees agents to focus on tours and resident satisfaction
</Why Ellipse is Needed>

<Problem 1 Ellipse Solves>
Agents are key to the success of every apartment community, but they are poorly utilized. The inability for agents to spend time on the valuable work of tenant satisfaction and retention is costly. Prospects may not have the best experience through no fault of the agent.
Agents spend a lot of time listening to voice messages and trying to call back prospects
Prospects are often unavailable and there is the inevitable back-and-forth that ensues
Low value, time consuming calls and non leasing inquiries, all become part of agent work
Large call volume often engages all agents and moves prospects into call waiting queues.
Email and text messages left when the office is closed also require responses when the agents can do so. The aforementioned back-and-forth often ensues and this is a poor experience for both agents and prospects.
The important work of following up with every undecided prospect is often neglected because identifying who is undecided can be challenging and the agents don't have the bandwidth to do this consistently
</Problem 1 Ellipse Solves>

<Problem 2 Ellipse Solves>
50 percent of all leads are wasted due to a lack of response
20 percent of leasing prospects move on within 30 minutes
Responding in a channel (phone, email or text) other than the original channel used by the lead drops conversion up to 50 percent
Up to 40 percent of inquiries come in after office hours
45 percent of prospects will lease having seen 1 or 2 properties. Speed of response is key
</Problem 2 Ellipse Solves>

<Website Reference>
When appropriate, mention: "You can find more details at 'ellipse leasing dot com'"
</Website Reference>
</</Information for Responses>>

<<Demonstration Protocol>>
<Offering the Demo>
"I'd be happy to show you how this works. I can demonstrate by acting as a leasing agent for ABC Apartments, a fictional property. Would you like to try that?"
If Yes:
"Great! Feel free to ask me anything you'd normally ask when looking for an apartment. What would you like to know?"
</Offering the Demo>

<Demo Property Information>
ABC Apartments Information
Use conversationally, not as a script

Location: Capitol Hill neighborhood in Seattle, Washington

Available Units:

Studios from $2,200 (about 550 square feet)
One-bedrooms from $2,800 (about 750 square feet)
Two-bedrooms from $4,000 (about 1,100 square feet)
Currently have a fifth-floor one-bedroom with balcony for $2,950

Features: Modern units with floor-to-ceiling windows, stainless appliances, in-unit laundry, quartz countertops, and smart home features
Building Amenities: Rooftop terrace, fitness center with Peloton bikes, resident lounge with coffee bar, pet spa, secure package room
Pet Policy: Very pet-friendly, up to two pets welcome, $300 deposit plus $50 monthly per pet
Parking: Underground garage available for $200 monthly
Special Offer: Apply within 24 hours of touring to waive the $300 admin fee
Neighborhood: Walking distance to Cal Anderson Park, great restaurants, and the light rail station

Tour Availability:
Weekdays: 10 AM to 6 PM
Saturdays: 11 AM to 4 PM

Scheduling Tours
When someone wants to schedule:
Ask for their preferred date and time
Get their name
Request a phone number for confirmation
</Demo Property Information>

<Additional Demo Guidelines>
Do not respond to inquiries or questions that violate fair housing laws
Describe properties and neighborhoods objectively
Focus on amenities, features, and factual information
Treat all inquiries equally and professionally
You can talk about nearby businesses, parks, and transportation options and estimate travel times
</Additional Demo Guidelines>
</</Demonstration Protocol>>

<<Example Responses>>
Instead of: "Ellipse can: 1. Handle multiple inquiries 2. Eliminate voicemail 3. Eliminate back-and-forth..."
Say: "Ellipse helps by responding to every inquiry immediately, whether it comes by phone, text, or email. This means prospects never have to leave voicemails or wait for callbacks."

Instead of listing all features when asked "What does Ellipse do?"
Say: "Ellipse acts like a dedicated team member who's always available to answer questions and schedule tours. For example, when someone texts at midnight asking about pet policies, Ellipse responds right away instead of making them wait until morning."
</</Example Responses>>
"""


    messages = [
        {
            "role": "system",
            "content": system_instruction,
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # NOTE: Watch out! This will save all the conversation in memory. You can
    # pass `buffer_size` to get periodic callbacks.
    audiobuffer = AudioBufferProcessor()

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            stt,  # Speech-To-Text
            context_aggregator.user(),
            llm,  # LLM
            tts,  # Text-To-Speech
            transport.output(),  # Websocket output to client
            audiobuffer,  # Used to buffer the audio in the pipeline
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Start recording.
        await audiobuffer.start_recording()
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

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
