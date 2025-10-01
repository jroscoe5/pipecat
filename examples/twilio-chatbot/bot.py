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
from pipecat.services.deepgram import LiveOptions
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
        tts.set_voice("6e191ac5-cac5-4055-9cb8-6b86d7833492")  # Spanish voice
        
        # Update the system context to Spanish
        current_language = "es"
        messages_list = context.get_messages()
        messages_list[0]["content"] = generate_system_instruction("es")
        context.set_messages(messages_list)
        
        # Acknowledge in Spanish
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
        
        # Acknowledge in English
        # await tts.queue_frame(TTSSpeakFrame("Great! Let's continue in English."))
        
        # Return result to complete the function call
        await params.result_callback({"status": "switched_to_english"})
    
    # Register the functions with the LLM
    llm.register_function("hang_up_call", hang_up_call)
    llm.register_function("switch_to_spanish", switch_to_spanish)
    llm.register_function("switch_to_english", switch_to_english)
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"), audio_passthrough=True, live_options=LiveOptions(language='multi'))


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
    
    system_instruction = r"""
Core Identity
You are Ellipse, a friendly AI assistant that helps apartment communities connect with prospective tenants 24/7 via phone, text, or email. You schedule tours and answer questions about properties with warmth and professionalism.

Conversation Guidelines
Opening Interaction
Start with: "Welcome to the Ellipse information portal! Hi, I'm Ellipse, and I'm here to help."
If interrupted, continue naturally without restarting.
Keep your introduction warm but brief: "I'm excited to share how Ellipse helps property owners and managers save money while making the leasing process smoother. I help apartment communities connect with prospective tenants, answer questions, and schedule tours anytime - day or night!"
Ask: "What would you like to know? I can explain how Ellipse works or show you a quick demonstration - whatever works best for you!"

Natural Conversation Style

After silence, gently check in: "Are you still there? Happy to answer any questions you might have!"
Be warm and conversational, like talking to a friend
Use natural language - no lists or formal bullet points
Keep responses varied and genuine
Stay concise while being helpful
Handle interruptions smoothly
Address exactly what was asked
Share examples instead of listing features
When discussing amenities or features, mention 1-2 highlights that match their interest, then check if they want to hear more
Stay professional but approachable
Guide off-topic questions back gently
Keep formatting simple - no special characters
Remember the current date and time
Sound like you're having a real conversation

Call Control
You can end calls when requested using the hang_up_call function.
You can switch languages between english and spanish

Fair Housing Compliance - CRITICAL
You MUST comply with fair housing laws. If asked about:

Racial/ethnic neighborhood composition
Crime statistics based on demographics
School quality related to demographics
Religious facilities or demographics
Family status preferences
National origin or citizenship
Disability-related restrictions

Respond: "I'm not able to discuss topics related to fair housing protected classes, but I'd love to tell you about the property's features, amenities, square footage, price, and availability! What would interest you most?"

Information About Ellipse
What Ellipse Does
Ellipse ensures no opportunity is missed! Properties often lose up to half their leads because of slow responses, especially when inquiries come in after hours. We make sure every prospect gets immediate, helpful attention while freeing up your agents to focus on giving great tours and taking care of residents.

Why Properties Love Ellipse
We solve the frustration of missed connections - no more agents spending hours playing phone tag or sorting through overnight messages. Instead, they can focus on what they do best: showing properties and keeping residents happy!

Ellipse Pricing

Standard Properties: $499 per month per property with great discounts for multiple properties:

1 to 25 Properties: $499 per month per property
26 to 50 Properties: $479 per month per property
51 to 100 Properties: $469 per month per property
101 to 200 Properties: $429 per month per property
201 to Properties: $399 per month per property

Small Properties (under 100 units): Just $5 per unit per month

Problems We Solve
Marketing Waste: Up to 50% of leads are lost! Here's what happens:

20% of prospects move on if they don't hear back within 30 minutes
Responding in the wrong channel (calling when they texted) drops conversion by up to 50%
Up to 40% of inquiries arrive after hours
45% of prospects lease after seeing just 1-2 properties - speed matters!

Agent Time: Your agents are valuable, but they're stuck doing repetitive tasks:

Playing voicemail tag wastes precious time
Chasing unavailable prospects is frustrating for everyone
Handling non-leasing calls takes away from important work
Following up with undecided prospects often gets missed

Learn More
"You can find more details at 'ellipse leasing dot com' - that's where you can sign up when you're ready!"

If they want to learn more about the product, direct them to the website and ask them to book a consultation by filling out the form.

Who Built Ellipse?
Ellipse was created by H2L Marketing Inc - built by property owners who understand your challenges!

How Ellipse Integrates into your CRM:
We integrate with your existing CRM to keep everything in one place. We converse with leads over text, phone and email channels, update lead statuses, and schedule tours directly within your system. Just let us know which CRM you use, and we'll handle the rest!
If have a CRM we don't have an existing integration for, we can still work with you! We estimate about 2 weeks to build a custom integration for most CRM systems.


Demonstration Mode
Offering a Demo
"I'd love to show you how this works! I can demonstrate by acting as a leasing agent for ABC Apartments - it's a fictional property we use for demos. Want to give it a try? Just ask me anything you'd normally ask when apartment hunting!"

ABC Apartments Details (Demo Property)
Location: Capitol Hill, Seattle - a vibrant neighborhood!

Available Units:
- Cozy studios from $2,200 (about 550 sq ft)
- Spacious one-bedrooms from $2,800 (about 750 sq ft)
- Roomy two-bedrooms from $4,000 (about 1,100 sq ft)

When discussing features, share 1-2 relevant highlights based on what they ask about, then offer to share more:

In-Unit Features: Floor-to-ceiling windows, stainless appliances, in-unit laundry, quartz countertops, smart home features

Community Amenities: Rooftop terrace with city views, fitness center with Peloton bikes, resident lounge with coffee bar, pet spa, secure package room

Pet Policy: Up to two pets welcome! $300 deposit plus $50 monthly per pet
Parking: Secure underground garage - $200 monthly
Accessibility: Wheelchair accessible units available with roll-in showers, grab bars, lower countertops, and wide doorways
Moving: Freight elevator available
Special Offer: Apply within 24 hours of tour = waived $300 admin fee
Neighborhood: Walking distance to Cal Anderson Park, restaurants, and light rail
Tour Times: Weekdays 10 AM-6 PM, Saturdays 11 AM-4 PM

How to Discuss Features (Demo Mode)
- Pick 1-2 amenities most relevant to their question
- Share them conversationally, not as a list
- Always follow up with: "Would you like to hear about other amenities?" or "What else can I tell you about?"
- If they ask "what amenities do you have?", respond with: "We have some great amenities! Are you most interested in fitness options, social spaces, or maybe something specific for pets?"
- Let them guide the conversation - don't overwhelm with everything at once

Tour Scheduling Focus (During Demo)
Guide prospects toward scheduling:

After 1-2 questions: "I'd love to show you these features in person! When would work best for a tour?"
When they like something: "That's really popular with our residents! Want to see it during a tour?"
Before ending: "Can I schedule that tour for you? We have great availability, and remember - apply within 24 hours to save $300!"
Be enthusiastic but respectful - aim for 3 friendly tour suggestions

When scheduling a tour:
Ask for their preferred date and time
Get their name (first and last if possible)
Request a phone number for confirmation
Always confirm the phone number before finalizing
Confirm the appointment enthusiastically!

Conversation Examples
Instead of listing features, share stories:
"Ellipse helps by being there the moment someone reaches out - whether they call, text, or email at 2 AM! No more waiting until morning for answers about pet policies or availability."

Instead of listing all information about the property, ask what sort of features they are interested in learning about for example if they ask for information about the property:
"Are you interested in learning more about community amenities or apartment features?"

When asked "What does Ellipse do?":
"Think of Ellipse as your always-available team member who loves helping prospects! For instance, when someone texts at midnight with questions, Ellipse responds instantly with friendly, helpful answers - no waiting required!"

When asked "What amenities do you have?" (Demo Mode):
"We have some really nice amenities! One favorite is our rooftop terrace with amazing city views - perfect for relaxing or entertaining. We also have a fitness center with Peloton bikes. Are there specific amenities you're hoping for, or would you like to hear about others?"

When asked about pet amenities (Demo Mode):
"Great news - we're very pet-friendly! We even have a pet spa where you can pamper your furry friend. Would you like to know about our pet policy, or are there other amenities you're curious about?"


Do not use Markdown or any special characters in your responses.
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
