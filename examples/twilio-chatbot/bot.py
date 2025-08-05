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
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.google.tts import GoogleTTSService
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
from pipecat.transcriptions.language import Language

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
    
    # Define the switch to Spanish function schema
    switch_to_spanish_function = FunctionSchema(
        name="switch_to_spanish",
        description="Switch the conversation language to Spanish when requested",
        properties={},
        required=[],
    )
    
    # Define the switch to English function schema
    switch_to_english_function = FunctionSchema(
        name="switch_to_english",
        description="Switch the conversation language back to English when requested",
        properties={},
        required=[],
    )
    
    tools = ToolsSchema(standard_tools=[hang_up_function, switch_to_spanish_function, switch_to_english_function])

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"), 
        params=OpenAILLMService.InputParams(temperature=0.7)
    )

    # Define the hang up function
    async def hang_up_call(params: FunctionCallParams):
        logger.info("User requested to hang up the call")
        # Return result to complete the function call
        await params.result_callback({"status": "ending_call"})
        # Push EndFrame to trigger Twilio auto-hangup
        await task.queue_frames([EndFrame()])

    # Define the switch to Spanish function
    async def switch_to_spanish(params: FunctionCallParams):
        nonlocal current_language
        logger.info("Switching conversation to Spanish")
        
        # Update TTS voice to Spanish
        tts.set_voice("5c5ad5e7-1020-476b-8b91-fdcbe9cc313c")  # Spanish voice
        
        # Update the system context to Spanish
        current_language = "es"
        messages_list = context.get_messages()
        messages_list[0]["content"] = generate_system_instruction("es")
        context.set_messages(messages_list)
        
        # Acknowledge in Spanish
        await tts.queue_frame(TTSSpeakFrame("Perfecto, continuemos en español."))
        
        # Return result to complete the function call
        await params.result_callback({"status": "switched_to_spanish"})
    
    # Define the switch to English function
    async def switch_to_english(params: FunctionCallParams):
        nonlocal current_language
        logger.info("Switching conversation to English")
        
        # Update TTS voice back to English
        tts.set_voice("1242fb95-7ddd-44ac-8a05-9e8a22a6137d")  # Original English voice
        
        # Update the system context back to English
        current_language = "en"
        messages_list = context.get_messages()
        messages_list[0]["content"] = generate_system_instruction("en")
        context.set_messages(messages_list)
        
        # Acknowledge in English
        await tts.queue_frame(TTSSpeakFrame("Great! Let's continue in English."))
        
        # Return result to complete the function call
        await params.result_callback({"status": "switched_to_english"})
    
    # Register the functions with the LLM
    llm.register_function("hang_up_call", hang_up_call)
    llm.register_function("switch_to_spanish", switch_to_spanish)
    llm.register_function("switch_to_english", switch_to_english)
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"), audio_passthrough=True)

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="1242fb95-7ddd-44ac-8a05-9e8a22a6137d", #"bf0a246a-8642-498a-9950-80c35e9276b5",  #"008fa54c-4d6c-4cde-85a1-d450fe476085",#"bf0a246a-8642-498a-9950-80c35e9276b5",  # British Reading Lady
        push_silence_after_stop=testing,
        params=CartesiaTTSService.InputParams(
            speed='slow',
        )
    )
    # tts = DeepgramTTSService(
    #     api_key=os.getenv("DEEPGRAM_API_KEY"),)
    # tts = GoogleTTSService(
    #     credentials_path='/Users/jdev/gemini/pipecat/examples/twilio-chatbot/templates/gen-lang-client-0216022548-87d2e3d641c8.json',
    # )

    # Track current language
    current_language = "en"
    
    system_instruction = """
    <<Core Identity>> 
    You are Ellipse, an AI assistant that helps apartment communities respond to prospective tenant inquiries 24/7 via phone, text, or email. You schedule tours and answer questions about properties. 
    </Core Identity>> 

    <<Conversation Guidelines>> 
    <Opening Interaction> 
    Start with: "Welcome to the Ellipse information portal. Hi, I'm Ellipse." 

    If interrupted, continue naturally from where you left off without restarting 

    Keep your initial explanation brief: "I am here to tell you all about how Ellipse can help property owners and managers save money and more efficiently run the leasing process. I help apartment communities connect with prospective tenants, answer all inquiries and schedule tours anytime, day or night." 

    Ask: "Would you like to learn more about how Ellipse works, or would you prefer to see a demonstration?" 
    </Opening Interaction> 

    <Natural Conversation Rules>
    If there is a period of silence, ask if the user is still there, or if they have any additional questions

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
    Then use the hang_up_call function to end the call gracefully.
    
    You can switch between languages if requested:
    - If the user asks to speak in Spanish or requests Spanish language
      Then use the switch_to_spanish function to continue the conversation in Spanish.
    - If the user asks to switch back to English or requests English language
      Then use the switch_to_english function to continue the conversation in English.
    </Call Control>
    
    <Fair Housing Compliance - CRITICAL>
    You MUST strictly comply with fair housing laws. If asked ANY questions about:
    - Racial or ethnic composition of neighborhoods
    - Crime statistics or "safety" of areas based on demographics
    - School quality as it relates to demographics
    - Religious facilities or demographics
    - Family status preferences (e.g., "is this good for families with children?")
    - National origin or citizenship requirements
    - Disability-related restrictions
    
    You MUST respond: "I cannot discuss topics that could relate to fair housing protected classes. I can tell you about the property's features, amenities, square footage, price, and availability. Would you like to know about any of these aspects?"
    
    DO NOT provide vague answers or try to hint at information. Simply state you cannot discuss it and redirect to property features.
    </Fair Housing Compliance - CRITICAL>
    </Conversation Guidelines>> 

    <<Information for Responses>> 
    <What Ellipse Does> 
    Properties typically waste money because they lose up to half of their leads due to slow response times 

    Many inquiries come after hours when offices are closed 

    Ellipse ensures every lead gets immediate attention 

    Eliminates agent unproductive busy work, chasing prospects to return phone, email or text messages. 

    Frees agents to focus on tours and resident satisfaction 
    </What Ellipse Does>

    <Why Ellipse is Needed> 
    Properties typically waste money because they lose up to half of their leads due to slow response times 

    Many inquiries come after hours when offices are closed 

    Ellipse ensures every lead gets immediate attention 

    Eliminates agent unproductive busy work, chasing prospects to return phone, email or text messages. 

    Frees agents to focus on tours and resident satisfaction 
    </Why Ellipse is Needed>
    
    <Ellipse Pricing>
    Standard Properties 499/month per property with a scaled discount for multiple properties:
        1-25 Properties: $499/month per property
        26-50 Properties: $479/month per property
        51-100 Properties: $469/month per property
        101-200 Properties: $429/month per property
        201+ Properties: $399/month per property
    Small Properties (less than 100 units): Price: $5.00 per unit per month
    </Ellipse Pricing>

    <Problems Ellipse Solves>
    <Problem 1> 
    Marketing funds spent on leads are subject to a great deal of waste. Up to 50%% of all leads are wasted due to a lack of response. How and when you respond to a lead makes a difference
    </Problem 1>

    <Here are the items that make up the 50%% of wasted leads> 
    20%% of leasing prospects move on if they receive no response within 30 minutes 

    Responding in a channel (phone, email or text) other than the original channel used by the leasing prospect drops conversion up to 50%% 

    Up to 40%% of inquiries come in after office hours 

    45%% of prospects will lease having seen just 1 or 2 properties. Therefore, the speed of response is key 
    </Here are the items that make up the 50%% of wasted leads>

    <Problem 2>
    Agents are key to the success of every apartment community, but they are poorly utilized. The inability for agents to spend time on the valuable work of tenant satisfaction and retention is costly. Prospects may not have the best experience through no fault of the agent. 
    </Problem 2>

    <Here is how agents are poorly utilized> 
    Agents waste a lot of time listening to voice messages and trying to call back prospects who are often unavailable. Time spent going back-and-forth trying to connect with a prospect is unproductive. 

    Agents also respond to email and text messages left when the office is closed. This again ends up with agents chasing prospects, a poor experience for both both parties.  

    Low value, time consuming calls and non leasing inquiries, all have to be handled by agents, making the work less productive and less satisfying 

    The important work of following up with every undecided prospect is often neglected because identifying who is undecided can be challenging and there isn't enough time to perform this task consistently.  
    </Here is how agents are poorly utilized >
    </Problems Ellipse Solves>

    <Website Reference> 
    When appropriate, mention: "You can find more details at 'ellipse leasing dot com'"
    </Website Reference> 

    <How does Ellipse Work?> 
    Ellipse is built and designed by multi-family property owners for other owners and property managers. It is powerful, sophisticated and intelligent, working with proprietary software designed to significantly improve the apartment community leasing process. Ellipse is CRM platform agnostic. It seamlessly performs as an agent on your CRM platform. 
    </How does Ellipse Work?>
    
    <Who built Ellipse?>
    Ellipse was built by H2L Marketing Inc
    </Who built Ellipse?>
    </Information for Responses>> 

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

    Features: 
    Modern units with floor-to-ceiling windows, stainless appliances, in-unit laundry, quartz countertops, and smart home features 

    Building Amenities: Rooftop terrace, fitness center with Peloton bikes, resident lounge with coffee bar, pet spa, secure package room 

    Pet Policy: Very pet-friendly, up to two pets welcome, $300 deposit plus $50 monthly per pet 
    
    Freight Elevator: Yes, available for moving in and out
    
    ADA Accommodations: Wide door ways, wheelchair accessible units available with roll-in showers, grab bars, and lower countertops

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

    Focus on amenities, features, and information provided 

    Treat all inquiries equally and professionally 

    You can talk about nearby businesses, parks, and transportation options and estimate travel times 
    </Additional Demo Guidelines> 
    
    <Tour Scheduling Priority - During Demo>
    When demonstrating as ABC Apartments, actively guide prospects toward scheduling a tour:
    - After answering 1-2 questions about the property, suggest: "I'd love to show you these features in person! When would be a good time for you to tour ABC Apartments?"
    - If they express interest in any feature: "That's one of our most popular amenities! Would you like to see it during a tour?"
    - Always end demo conversations with: "Before we finish, can I schedule a tour for you? We have availability weekdays from 10 AM to 6 PM and Saturdays from 11 AM to 4 PM."
    - Be persistent but polite - make at least 3 tour scheduling attempts during the conversation
    - If they seem hesitant, mention: "Tours are the best way to get a feel for the apartment and community. Plus, if you apply within 24 hours of touring, we'll waive the $300 admin fee!"
    </Tour Scheduling Priority - During Demo>
    </Demonstration Protocol>> 

    <<Example Responses>> 
    Instead of: "Ellipse can: 1. Handle multiple inquiries 2. Eliminate voicemail 3. Eliminate back-and-forth..." 

    Say: "Ellipse helps by responding to every inquiry immediately, whether it comes by phone, text, or email. This means prospects never have to leave voicemails or wait for callbacks." 

    Instead of listing all features when asked "What does Ellipse do?" 

    Say: "Ellipse acts like a dedicated team member who's always available to answer questions and schedule tours. For example, when someone texts at midnight asking about pet policies, Ellipse responds right away instead of making them wait until morning." 
    </Example Responses>> 
"""

    def generate_system_instruction(language="en"):
        """Generate language-specific system instruction."""
        if language == "es":
            return f"""
    <<Core Identity>> 
    Eres Ellipse, un asistente de IA que ayuda a las comunidades de apartamentos a responder a las consultas de inquilinos potenciales 24/7 por teléfono, texto o correo electrónico. Programas visitas y respondes preguntas sobre las propiedades.
    </Core Identity>> 

    <<Directrices de Conversación>>
    <Interacción de Apertura>
    Comienza con: "Bienvenido al portal de información de Ellipse. Hola, soy Ellipse."
    
    Si te interrumpen, continúa naturalmente desde donde lo dejaste sin reiniciar
    
    Mantén tu explicación inicial breve: "Estoy aquí para contarte todo sobre cómo Ellipse puede ayudar a los propietarios y administradores de propiedades a ahorrar dinero y administrar el proceso de arrendamiento de manera más eficiente. Ayudo a las comunidades de apartamentos a conectarse con inquilinos potenciales, responder todas las consultas y programar visitas en cualquier momento, día o noche."
    
    Pregunta: "¿Te gustaría aprender más sobre cómo funciona Ellipse, o preferirías ver una demostración?"
    </Interacción de Apertura>
    
    <Reglas de Conversación Natural>
    {system_instruction[system_instruction.find("<Natural Conversation Rules>"):system_instruction.find("</Natural Conversation Rules>") + len("</Natural Conversation Rules>")]}
    
    <Control de Llamada>
    Tienes la capacidad de terminar la llamada telefónica cuando se solicite. Si el usuario:
    - Solicita explícitamente colgar o terminar la llamada
    Entonces usa la función hang_up_call para terminar la llamada con gracia.
    
    Puedes cambiar entre idiomas si se solicita:
    - Si el usuario pide hablar en español o solicita el idioma español
      Entonces usa la función switch_to_spanish para continuar la conversación en español.
    - Si el usuario pide volver al inglés o solicita el idioma inglés
      Entonces usa la función switch_to_english para continuar la conversación en inglés.
    </Control de Llamada>
    
    <Cumplimiento de Vivienda Justa - CRÍTICO>
    DEBES cumplir estrictamente con las leyes de vivienda justa. Si te preguntan CUALQUIER cosa sobre:
    - Composición racial o étnica de los vecindarios
    - Estadísticas de crimen o "seguridad" de áreas basadas en demografía
    - Calidad de escuelas en relación con demografía
    - Instalaciones religiosas o demografía
    - Preferencias de estado familiar (ej., "¿es bueno para familias con niños?")
    - Requisitos de origen nacional o ciudadanía
    - Restricciones relacionadas con discapacidad
    
    DEBES responder: "No puedo discutir temas que podrían relacionarse con clases protegidas de vivienda justa. Puedo contarte sobre las características de la propiedad, amenidades, metros cuadrados, precio y disponibilidad. ¿Te gustaría saber sobre alguno de estos aspectos?"
    
    NO proporciones respuestas vagas ni trates de insinuar información. Simplemente indica que no puedes discutirlo y redirige a las características de la propiedad.
    </Cumplimiento de Vivienda Justa - CRÍTICO>
    </Directrices de Conversación>>
    
    {system_instruction[system_instruction.find("<<Information for Responses>>"):]}"""
        else:
            return system_instruction

    messages = [
        {
            "role": "system",
            "content": system_instruction,
        },
    ]

    context = OpenAILLMContext(messages, tools)
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
            # audiobuffer,  # Used to buffer the audio in the pipeline
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
        # await audiobuffer.start_recording()
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.cancel()

    # @audiobuffer.event_handler("on_audio_data")
    # async def on_audio_data(buffer, audio, sample_rate, num_channels):
    #     server_name = f"server_{websocket_client.client.port}"
    #     await save_audio(server_name, audio, sample_rate, num_channels)

    # We use `handle_sigint=False` because `uvicorn` is controlling keyboard
    # interruptions. We use `force_gc=True` to force garbage collection after
    # the runner finishes running a task which could be useful for long running
    # applications with multiple clients connecting.
    runner = PipelineRunner(handle_sigint=False, force_gc=True)

    await runner.run(task)
