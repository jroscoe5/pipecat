# Twilio + Google Gemini Flash Live Phone Bot

This example demonstrates how to create a real-time phone conversation bot using Twilio for telephony and Google's Gemini Flash Live model for natural conversation.

## Features

- Real-time phone conversations with Google Gemini Flash Live
- Low-latency audio streaming via Twilio
- Automatic speech recognition and synthesis handled by Gemini
- Audio recording capability for conversation history

## Prerequisites

1. **Google Cloud Account**
   - Enable the Gemini API
   - Create an API key with access to Gemini models

2. **Twilio Account**
   - Sign up at https://www.twilio.com
   - Get your Account SID and Auth Token
   - Purchase a phone number

3. **Public Server or ngrok**
   - Your server needs to be accessible from the internet
   - For local development, use ngrok: `ngrok http 8765`

## Setup Instructions

### 1. Install Dependencies

```bash
cd examples/twilio-gemini-live
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp env.example .env
```

Edit `.env` and add:
- `GOOGLE_API_KEY`: Your Google API key
- `TWILIO_ACCOUNT_SID`: Your Twilio Account SID
- `TWILIO_AUTH_TOKEN`: Your Twilio Auth Token

### 3. Update the TwiML Template

Edit `templates/streams.xml` and replace `<your server url>` with your public server URL:

```xml
<Stream url="wss://your-domain.com/ws"></Stream>
```

For ngrok, it would look like:
```xml
<Stream url="wss://abc123.ngrok.io/ws"></Stream>
```

### 4. Configure Twilio Phone Number

In your Twilio Console:
1. Go to Phone Numbers > Manage > Active Numbers
2. Click on your phone number
3. In the Voice Configuration section:
   - Set "A call comes in" to: Webhook
   - URL: `https://your-domain.com/` (or your ngrok URL)
   - HTTP Method: POST

### 5. Run the Server

```bash
python server.py
```

The server will start on port 8765.

### 6. Test the Integration

Call your Twilio phone number. You should hear Gemini greet you and you can have a natural conversation!

## How It Works

1. **Incoming Call**: When someone calls your Twilio number, Twilio makes a POST request to your server
2. **TwiML Response**: Your server responds with TwiML that tells Twilio to stream audio via WebSocket
3. **WebSocket Connection**: Twilio establishes a WebSocket connection and streams audio in real-time
4. **Gemini Processing**: The audio is sent to Gemini Multimodal Live, which handles:
   - Speech recognition (audio to text)
   - Language understanding and response generation
   - Text-to-speech (response to audio)
5. **Audio Response**: Gemini's audio response is streamed back through Twilio to the caller

## Customization

### Change Gemini's Voice

In `bot.py`, modify the `voice_id` parameter:
```python
voice_id="Puck",  # Options: Aoede, Charon, Fenrir, Kore, Puck
```

### Adjust the System Prompt

Modify the `system_instruction` in `bot.py` to change Gemini's behavior:
```python
system_instruction = """
Your custom instructions here...
"""
```

### Use Different Gemini Models

Change the model parameter (defaults to Flash for lower latency):
```python
model="gemini-2.0-flash-exp",  # or "gemini-2.0-pro-exp" for more capabilities
```

## Troubleshooting

1. **No audio on call**: Check that your server URL in `streams.xml` uses `wss://` (secure WebSocket)
2. **Connection drops**: Ensure your firewall allows WebSocket connections
3. **High latency**: Consider using a server closer to your location
4. **API errors**: Verify your Google API key has access to Gemini models

## Cost Considerations

- **Twilio**: Charges for phone numbers (~$1/month) and per-minute usage
- **Google Gemini**: Charges based on audio duration and model usage
- **Server**: Your hosting costs

## Security Notes

- Never commit your `.env` file with real credentials
- Use HTTPS/WSS for all connections
- Consider implementing authentication for your webhook endpoint
- Rotate your API keys regularly