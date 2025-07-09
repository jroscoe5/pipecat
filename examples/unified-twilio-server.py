#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import json
import importlib.util
import sys
import os

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse
from dotenv import load_dotenv
from loguru import logger

# Load .env file from the examples directory
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(env_path, override=True)

# Configure logger once for the entire application
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Dynamically import the bot modules
def import_bot_module(path, module_name):
    # Temporarily suppress logger.remove() calls during import
    original_remove = logger.remove
    logger.remove = lambda x: None
    
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        # Restore original logger.remove
        logger.remove = original_remove

# Import the bot functions from both examples
openai_bot = import_bot_module(os.path.join(script_dir, 'twilio-chatbot', 'bot.py'), 'openai_bot')
gemini_bot = import_bot_module(os.path.join(script_dir, 'twilio-gemini-live', 'bot.py'), 'gemini_bot')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Get the server URL from environment or use default
SERVER_URL = os.getenv("SERVER_URL", "wss://your-ngrok-url.ngrok-free.app")
# Ensure it's using wss:// for WebSocket
if SERVER_URL.startswith("https://"):
    SERVER_URL = SERVER_URL.replace("https://", "wss://")
print(f"Loaded SERVER_URL: {SERVER_URL}")

# OpenAI bot endpoints
@app.post("/openai")
async def start_openai_call():
    print("POST TwiML for OpenAI bot")
    print(f"Using SERVER_URL: {SERVER_URL}")
    # Return TwiML that points to the OpenAI WebSocket endpoint
    twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{SERVER_URL}/ws/openai"></Stream>
  </Connect>
  <Pause length="40"/>
</Response>'''
    print(f"Returning TwiML: {twiml}")
    return HTMLResponse(content=twiml, media_type="application/xml")


@app.websocket("/ws/openai")
async def openai_websocket_endpoint(websocket: WebSocket):
    print("OpenAI WebSocket connection attempt")
    await websocket.accept()
    print("OpenAI WebSocket accepted")
    start_data = websocket.iter_text()
    await start_data.__anext__()
    call_data = json.loads(await start_data.__anext__())
    print(f"OpenAI Bot - Call Data: {call_data}", flush=True)
    stream_sid = call_data["start"]["streamSid"]
    call_sid = call_data["start"]["callSid"]
    print("OpenAI Bot - WebSocket connection accepted")
    await openai_bot.run_bot(websocket, stream_sid, call_sid, app.state.testing)


# Gemini bot endpoints
@app.post("/gemini")
async def start_gemini_call():
    print("POST TwiML for Gemini bot")
    print(f"Using SERVER_URL: {SERVER_URL}")
    # Return TwiML that points to the Gemini WebSocket endpoint
    twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{SERVER_URL}/ws/gemini"></Stream>
  </Connect>
  <Pause length="40"/>
</Response>'''
    print(f"Returning TwiML: {twiml}")
    return HTMLResponse(content=twiml, media_type="application/xml")


@app.websocket("/ws/gemini")
async def gemini_websocket_endpoint(websocket: WebSocket):
    print("Gemini WebSocket connection attempt")
    await websocket.accept()
    print("Gemini WebSocket accepted")
    start_data = websocket.iter_text()
    await start_data.__anext__()
    call_data = json.loads(await start_data.__anext__())
    print(f"Gemini Bot - Call Data: {call_data}", flush=True)
    stream_sid = call_data["start"]["streamSid"]
    call_sid = call_data["start"]["callSid"]
    print("Gemini Bot - WebSocket connection accepted")
    await gemini_bot.run_bot(websocket, stream_sid, call_sid, app.state.testing)


# Root endpoint for health checks
@app.get("/")
async def root():
    return {"status": "healthy", "bots": ["openai", "gemini"], "server_url": SERVER_URL}

# Test endpoint to return simple TwiML
@app.post("/test")
async def test_twiml():
    print("Test TwiML endpoint called")
    twiml = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Hello from the unified server. This is a test message.</Say>
  <Pause length="2"/>
  <Hangup/>
</Response>'''
    return HTMLResponse(content=twiml, media_type="application/xml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Pipecat Twilio Server")
    parser.add_argument(
        "-t", "--test", action="store_true", default=False, help="set the server in testing mode"
    )
    args, _ = parser.parse_known_args()

    app.state.testing = args.test

    print("Starting unified server...")
    print(f"Server URL configured as: {SERVER_URL}")
    print("\nConfigure your Twilio phone numbers:")
    print(f"  OpenAI bot webhook: {SERVER_URL.replace('wss://', 'https://')}/openai")
    print(f"  Gemini bot webhook: {SERVER_URL.replace('wss://', 'https://')}/gemini")
    print("\nWebSocket endpoints:")
    print(f"  OpenAI: {SERVER_URL}/ws/openai")
    print(f"  Gemini: {SERVER_URL}/ws/gemini")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)