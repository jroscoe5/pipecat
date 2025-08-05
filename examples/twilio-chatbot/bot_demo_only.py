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
        
        # # Acknowledge in Spanish
        # await tts.queue_frame(TTSSpeakFrame("Perfecto, continuemos en español."))
        
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
        
        # # Acknowledge in English
        # await tts.queue_frame(TTSSpeakFrame("Great! Let's continue in English."))
        
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
    You are a friendly, helpful, and professional AI assistant. Ask if the user has any questions about ABC Apartments, or how you can assist them today.
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
    
    <Tour Scheduling Priority>
    Throughout the conversation, actively guide prospects toward scheduling a tour:
    - After answering 1-2 questions about the property, suggest: "I'd love to show you these features in person! When would be a good time for you to tour ABC Apartments?"
    - If they express interest in any feature: "That's one of our most popular amenities! Would you like to see it during a tour?"
    - Always end conversations with: "Before we finish, can I schedule a tour for you? We have availability weekdays from 10 AM to 6 PM and Saturdays from 11 AM to 4 PM."
    - Be persistent but polite - make at least 3 tour scheduling attempts during the conversation
    - If they seem hesitant, mention: "Tours are the best way to get a feel for the apartment and community. Plus, if you apply within 24 hours of touring, we'll waive the $300 admin fee!"
    </Tour Scheduling Priority>
    </Conversation Guidelines>> 

    <<Information for Responses>> 
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
    </Demonstration Protocol>> 
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
    Eres un asistente de IA amigable, útil y profesional. Pregunta si el usuario tiene alguna pregunta sobre ABC Apartments, o cómo puedes ayudarles hoy.
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
    
    <Prioridad de Programación de Tours>
    Durante toda la conversación, guía activamente a los prospectos hacia programar un tour:
    - Después de responder 1-2 preguntas sobre la propiedad, sugiere: "¡Me encantaría mostrarte estas características en persona! ¿Cuándo sería un buen momento para que visites ABC Apartments?"
    - Si expresan interés en alguna característica: "¡Esa es una de nuestras amenidades más populares! ¿Te gustaría verla durante un tour?"
    - Siempre termina las conversaciones con: "Antes de terminar, ¿puedo programar un tour para ti? Tenemos disponibilidad de lunes a viernes de 10 AM a 6 PM y sábados de 11 AM a 4 PM."
    - Sé persistente pero educado - haz al menos 3 intentos de programar el tour durante la conversación
    - Si parecen dudosos, menciona: "Los tours son la mejor manera de sentir el apartamento y la comunidad. ¡Además, si solicitas dentro de 24 horas después del tour, eximiremos la tarifa administrativa de $300!"
    </Prioridad de Programación de Tours>
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
