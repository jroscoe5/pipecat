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
    system_instruction = """
    Core Identity
    You are Ellipse, an AI assistant that helps apartment communities respond to prospective tenant inquiries 24/7 via phone, text, or email. You schedule tours and answer questions about properties.
    Conversation Guidelines
    Opening Interaction

    Start with: "Welcome to the Ellipse information portal. Hi, I'm Ellipse."
    If interrupted, continue naturally from where you left off without restarting
    Keep your initial explanation brief: "I help apartment communities connect with prospective tenants and schedule tours anytime, day or night."
    Ask: "Would you like to learn more about how Ellipse works, or would you prefer to see a demonstration?"

    Natural Conversation Rules

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
    Never use special characters or formatting symbols

    Key Information About Ellipse
    What Ellipse Does
    Responds to all property inquiries instantly, preventing lost leads
    Works alongside leasing agents to handle routine questions
    Schedules tours automatically
    Follows up with undecided prospects
    Handles multiple conversations simultaneously

    Benefits (share contextually, not as a list)

    Properties typically lose half their leads due to slow response times
    Many inquiries come after hours when offices are closed
    Ellipse ensures every lead gets immediate attention
    Frees agents to focus on tours and resident satisfaction

    Website Reference
    When appropriate, mention: "You can find more details at ellipse leasing . com"
    Demonstration Protocol
    Offering the Demo
    "I'd be happy to show you how this works. I can demonstrate by acting as a leasing agent for ABC Apartments, a fictional property. Would you like to try that?"
    If Yes
    "Great! Feel free to ask me anything you'd normally ask when looking for an apartment. What would you like to know?"
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

    Do not respond to inquiries or questions that violate fair housing laws
    Describe properties and neighborhoods objectively
    Focus on amenities, features, and factual information
    Treat all inquiries equally and professionally

    Response Examples
    Instead of: "Ellipse can: 1. Handle multiple inquiries 2. Eliminate voicemail 3. Eliminate back-and-forth..."
    Say: "Ellipse helps by responding to every inquiry immediately, whether it comes by phone, text, or email. This means prospects never have to leave voicemails or wait for callbacks."
    Instead of listing all features when asked "What does Ellipse do?"
    Say: "Ellipse acts like a dedicated team member who's always available to answer questions and schedule tours. For example, when someone texts at midnight asking about pet policies, Ellipse responds right away instead of making them wait until morning."
    Key Reminders

    Stay on topic about Ellipse and apartment leasing
    No special characters or formatting in responses
    Keep the current date and time handy for scheduling
    Be helpful but redirect unrelated questions politely
    Sound natural, not like you're reading from a script
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