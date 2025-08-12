import logging
from typing import Type
import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    TerminationEvent,
    TurnEvent,
)
from dotenv import load_dotenv
import os
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

TWILIO_SAMPLE_RATE = 8000  # Twilio's sample rate
BYTES_PER_MS = 8  # For 8kHz μ-law (1 byte per sample per channel)

class TwilioTranscriber:
    def __init__(self):
        self.client = StreamingClient(
            StreamingClientOptions(
                api_key=os.getenv('ASSEMBLYAI_API_KEY'),
                api_host="streaming.assemblyai.com",
            )
        )
        self.audio_buffer = bytearray()
        self.chunk_size_ms = 100  # Send 100ms chunks
        self.min_chunk_size = self.chunk_size_ms * BYTES_PER_MS
        
        # Event handlers
        self.client.on(StreamingEvents.Begin, self.on_begin)
        self.client.on(StreamingEvents.Turn, self.on_turn)
        self.client.on(StreamingEvents.Termination, self.on_terminated)
        self.client.on(StreamingEvents.Error, self.on_error)

        self.last_print_time = time.time()
        self.active = False

    def on_begin(self, _: Type[StreamingClient], event: BeginEvent):
        logger.info(f"Session started: {event.id}")
        self.active = True

    def on_turn(self, _: Type[StreamingClient], event: TurnEvent):
        if not event.transcript:
            return

        # Clear the current line
        print('\r', end='', flush=True)
        
        if event.end_of_turn:
            # Final transcript
            print(f"✅ {event.transcript}\n", flush=True)
        else:
            # Partial transcript (update in-place)
            print(f"⌛ {event.transcript}", end='', flush=True)

    def on_terminated(self, _: Type[StreamingClient], event: TerminationEvent):
        logger.info(f"Session ended after {event.audio_duration_seconds:.2f}s")
        self.active = False

    def on_error(self, _: Type[StreamingClient], error: StreamingError):
        logger.error(f"Error: {error}")
        self.active = False

    def connect(self):
        try:
            self.client.connect(
                StreamingParameters(
                    sample_rate=TWILIO_SAMPLE_RATE,
                    encoding="pcm_mulaw",
                    format_turns=True,
                )
            )
            # self.client.start()  # <== ADD THIS LINE
            logger.info("Connected to AssemblyAI")
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

    def stream(self, audio_data: bytes):
        """Buffer and stream audio to AssemblyAI"""
        if not self.active:
            return

        self.audio_buffer.extend(audio_data)
        logger.debug(f"Received {len(audio_data)} bytes (Total buffer: {len(self.audio_buffer)} bytes)")

        # Send chunks when enough data is available
        while len(self.audio_buffer) >= self.min_chunk_size:
            chunk = bytes(self.audio_buffer[:self.min_chunk_size])
            try:
                self.client.stream(chunk)
                self.audio_buffer = self.audio_buffer[self.min_chunk_size:]
                logger.debug(f"Sent {len(chunk)} bytes")
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                break

    def close(self):
        """Flush buffer and close connection"""
        if len(self.audio_buffer) > 0:
            logger.info(f"Flushing {len(self.audio_buffer)} remaining bytes")
            self.client.stream(bytes(self.audio_buffer))
        
        self.client.disconnect(terminate=True)
        logger.info("Disconnected from AssemblyAI")