import asyncio
import base64
import json
import uuid
from typing import Optional

import aiohttp
import websockets
from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import StartFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.heygen.api import HeyGenApi, HeyGenSession, NewSessionRequest
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.asyncio.task_manager import BaseTaskManager

HEY_GEN_SAMPLE_RATE = 24000


class HeyGenClient:
    def __init__(
        self,
        *,
        api_key: str,
        session: aiohttp.ClientSession,
        params: TransportParams,
    ) -> None:
        self._api = HeyGenApi(api_key, session=session)
        self._heyGen_session: Optional[HeyGenSession] = None
        self._websocket = None
        self._task_manager: Optional[BaseTaskManager] = None
        self._params = params
        self._in_sample_rate = 0
        self._out_sample_rate = 0
        self._connected = False
        self._resampler = create_stream_resampler()
        self._buffered_audio_duration_ms = 0

    async def _initialize(self):
        self._heyGen_session = await self._api.new_session(
            # TODO: we should receive this as parameter
            NewSessionRequest(
                avatarName="Shawn_Therapist_public",
                version="v2",
            )
        )
        logger.debug(f"HeyGen session: {self._heyGen_session}")
        await self._api.start_session(self._heyGen_session.session_id)
        logger.info("HeyGen session started")

    async def setup(self, setup: FrameProcessorSetup):
        """Setup the client and initialize the conversation.

        Args:
            setup: The frame processor setup configuration.
        """
        if self._heyGen_session is not None:
            logger.debug("heygen_session already initialized")
            return

        self._task_manager = setup.task_manager

        try:
            await self._initialize()
            # TODO: implement the rest of the logic
        except Exception as e:
            logger.error(f"Failed to setup HeyGenClient: {e}")
            await self.cleanup()

    async def cleanup(self):
        """Cleanup client resources."""
        try:
            if self._heyGen_session is not None:
                await self._api.close_session(self._heyGen_session.session_id)
                self._heyGen_session = None
                self._connected = False
                self._buffered_audio_duration_ms = 0
        except Exception as e:
            logger.exception(f"Exception during cleanup: {e}")

    async def start(self, frame: StartFrame):
        logger.info(f"HeyGenClient starting")
        self._in_sample_rate = self._params.audio_in_sample_rate or frame.audio_in_sample_rate
        self._out_sample_rate = self._params.audio_out_sample_rate or frame.audio_out_sample_rate
        await self._ws_connect()

    async def stop(self):
        logger.info(f"HeyGenVideoService stopping")
        await self._ws_disconnect()
        await self.cleanup()

    # websocket connection methods
    async def _ws_connect(self):
        """Connect to HeyGen websocket endpoint"""
        try:
            if self._websocket:
                logger.debug(f"HeyGenClient ws already connected!")
                return
            logger.debug(f"HeyGenClient ws connecting")
            self._websocket = await websockets.connect(
                uri=self._heyGen_session.realtime_endpoint,
            )
            self._connected = True
            self._receive_task = self._task_manager.create_task(self._ws_receive_task_handler())
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None

    async def _ws_receive_task_handler(self):
        """Handle incoming WebSocket messages."""
        while self._connected:
            try:
                message = await asyncio.wait_for(self._websocket.recv(), timeout=1.0)
                parsed_message = json.loads(message)
                await self._handle_ws_server_event(parsed_message)
            except asyncio.TimeoutError:
                self._task_manager.task_reset_watchdog()
            except websockets.exceptions.ConnectionClosedOK:
                break
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                break

    async def _handle_ws_server_event(self, event: dict) -> None:
        """Handle an event from HeyGen websocket"""
        event_type = event.get("type")
        if event_type == "agent.status":
            logger.info(f"HeyGenClient ws received agent status: {event}")
        else:
            logger.error(f"HeyGenClient ws received unknown event: {event_type}")

    async def _ws_disconnect(self) -> None:
        """Disconnect from HeyGen websocket endpoint"""
        try:
            if self._websocket:
                await self._websocket.close()
            self._connected = False
        except Exception as e:
            logger.error(f"{self} disconnect error: {e}")
        finally:
            self._websocket = None

    async def _ws_send(self, message: dict) -> None:
        """Send a message to HeyGen websocket"""
        try:
            if self._websocket:
                await self._websocket.send(json.dumps(message))
            else:
                logger.error(f"{self} websocket not connected")
        except Exception as e:
            logger.error(f"Error sending message to HeyGen websocket: {e}")
            raise e

    async def interrupt(self) -> None:
        """Stops the avatar’s current task and resets it to an idle animation."""
        await self._ws_send(
            {
                "type": "agent.interrupt",
                "event_id": str(uuid.uuid4()),
            }
        )

    async def start_agent_listening(self) -> None:
        """Triggers the avatar's listening animation."""
        await self._ws_send(
            {
                "type": "agent.start_listening",
                "event_id": str(uuid.uuid4()),
            }
        )

    async def stop_agent_listening(self) -> None:
        """Stop listening animation"""
        await self._ws_send(
            {
                "type": "agent.stop_listening",
                "event_id": str(uuid.uuid4()),
            }
        )

    @property
    def out_sample_rate(self) -> int:
        """Get the output sample rate.

        Returns:
            The output sample rate in Hz.
        """
        return self._out_sample_rate

    @property
    def in_sample_rate(self) -> int:
        """Get the input sample rate.

        Returns:
            The input sample rate in Hz.
        """
        return self._in_sample_rate

    async def send_audio(
        self, audio: bytes, sample_rate: int, event_id: str, finish: bool = False
    ) -> None:
        audio = await self._resampler.resample(audio, sample_rate, HEY_GEN_SAMPLE_RATE)
        self._buffered_audio_duration_ms += self._calculate_audio_duration_ms(
            audio, HEY_GEN_SAMPLE_RATE
        )
        await self._agent_audio_buffer_append(audio, event_id)

        if finish and self._buffered_audio_duration_ms < 80:
            await self._agent_audio_buffer_clear()
            self._buffered_audio_duration_ms = 0

        if finish or self._buffered_audio_duration_ms > 1000:
            logger.info(
                f"Audio buffer duration from buffer: {self._buffered_audio_duration_ms:.2f}ms"
            )
            await self._agent_audio_buffer_commit(event_id)
            self._buffered_audio_duration_ms = 0

    def _calculate_audio_duration_ms(self, audio: bytes, sample_rate: int) -> float:
        # Each sample is 2 bytes (16-bit audio)
        num_samples = len(audio) / 2
        return (num_samples / sample_rate) * 1000

    async def _agent_audio_buffer_append(self, audio: bytes, event_id: str) -> None:
        audio_base64 = base64.b64encode(audio).decode("utf-8")
        await self._ws_send(
            {
                "type": "agent.audio_buffer_append",
                "audio": audio_base64,
                "event_id": str(uuid.uuid4()),
            }
        )

    async def _agent_audio_buffer_clear(self) -> None:
        await self._ws_send(
            {
                "type": "agent.audio_buffer_clear",
                "event_id": str(uuid.uuid4()),
            }
        )

    # TODO: should test it later to always sending the audio commiting it, instead of buffering it
    async def _agent_audio_buffer_commit(self, event_id: str) -> None:
        audio_base64 = base64.b64encode(b"\x00").decode("utf-8")
        await self._ws_send(
            {
                "type": "agent.audio_buffer_commit",
                "audio": audio_base64,
                "event_id": str(uuid.uuid4()),
            }
        )
