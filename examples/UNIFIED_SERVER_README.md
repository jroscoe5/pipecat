# Unified Twilio Server for Multiple Bots

This setup allows you to run multiple Twilio bots (OpenAI and Gemini) behind a single ngrok endpoint using path-based routing.

## Architecture

```
                    ┌─────────────┐
                    │   Twilio    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │    ngrok    │
                    │ (single URL)│
                    └──────┬──────┘
                           │
                ┌──────────┴──────────┐
                │                     │
         POST /openai          POST /gemini
                │                     │
         WS /ws/openai         WS /ws/gemini
                │                     │
        ┌───────▼──────┐      ┌──────▼───────┐
        │  OpenAI Bot  │      │  Gemini Bot  │
        │   (GPT-4)    │      │ (Flash Live) │
        └──────────────┘      └──────────────┘
```

## Setup Instructions

### 1. Install Dependencies

Make sure you have all dependencies for both bots:

```bash
cd examples
pip install -r twilio-chatbot/requirements.txt
pip install -r twilio-gemini-live/requirements.txt
pip install python-dotenv
```

### 2. Configure Environment Variables

Create a `.env` file in the examples directory with all required credentials:

```bash
# Server URL (will be updated with your ngrok URL)
SERVER_URL=wss://your-ngrok-url.ngrok-free.app

# For OpenAI Bot
OPENAI_API_KEY=your_openai_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
CARTESIA_API_KEY=your_cartesia_api_key

# For Gemini Bot
GOOGLE_API_KEY=your_google_api_key

# Twilio Credentials (shared by both bots)
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
```

### 3. Start ngrok

In a terminal, start ngrok to create a public URL:

```bash
ngrok http 8765
```

Copy the HTTPS URL provided by ngrok (e.g., `https://abc123.ngrok-free.app`)

### 4. Update the Server URL

Update the `SERVER_URL` in your `.env` file with your ngrok URL (change https to wss):

```bash
SERVER_URL=wss://abc123.ngrok-free.app
```

### 5. Run the Unified Server

```bash
cd examples
python unified-twilio-server.py
```

The server will display:
- OpenAI bot webhook URL
- Gemini bot webhook URL
- WebSocket endpoints for both bots

### 6. Configure Twilio Phone Numbers

You have two options:

#### Option A: Use Different Phone Numbers
1. Configure one phone number for the OpenAI bot:
   - Webhook URL: `https://your-ngrok-url.ngrok-free.app/openai`
   
2. Configure another phone number for the Gemini bot:
   - Webhook URL: `https://your-ngrok-url.ngrok-free.app/gemini`

#### Option B: Use One Phone Number with Call Routing
You could implement additional logic to route calls based on:
- Caller ID
- Time of day
- IVR menu selection
- Random selection for A/B testing

### 7. Test Your Bots

- Call the OpenAI bot phone number to chat with GPT-4
- Call the Gemini bot phone number to chat with Gemini Flash Live

## How It Works

1. **Incoming Call**: Twilio receives a call and makes a POST request to your webhook URL
2. **Path Routing**: The unified server routes to different handlers based on the path:
   - `/openai` → OpenAI bot TwiML response
   - `/gemini` → Gemini bot TwiML response
3. **WebSocket Streams**: Each bot gets its own WebSocket path:
   - `/ws/openai` → OpenAI bot WebSocket handler
   - `/ws/gemini` → Gemini bot WebSocket handler
4. **Bot Processing**: Each bot runs independently with its own pipeline and services

## Advantages

- **Single ngrok URL**: No need for multiple ngrok instances
- **Easy Management**: All bots run from one server process
- **Cost Effective**: Single server deployment
- **Flexible Routing**: Easy to add more bots or routing logic

## Adding More Bots

To add another bot:

1. Create your bot module in a new directory
2. Import it in `unified-twilio-server.py`
3. Add new endpoints for your bot (e.g., `/mynewbot` and `/ws/mynewbot`)
4. Configure a Twilio phone number to use the new webhook

## Troubleshooting

1. **Import Errors**: Make sure the bot directories (`twilio-chatbot`, `twilio-gemini-live`) exist
2. **WebSocket Connection Failed**: Ensure your ngrok URL uses `wss://` in the `.env` file
3. **Bot Not Responding**: Check the console for specific bot errors
4. **Wrong Bot Answering**: Verify the webhook URLs in Twilio match the correct paths

## Production Considerations

For production deployment:
- Use a proper domain with SSL instead of ngrok
- Implement authentication on webhook endpoints
- Add health checks and monitoring
- Consider load balancing for high traffic
- Use environment-specific configuration files