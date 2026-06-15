"""60dB speech-to-text for LiveKit Agents.

Implemented as a *non-streaming* recognizer. ``VoicePipelineAgent`` will
automatically wrap this with its VAD-driven ``StreamAdapter``, so we only
need to transcribe a finished audio buffer - no socket lifecycle to manage.
This is the same approach the OpenAI Whisper plugin uses.
"""

from __future__ import annotations

import io
import wave

from livekit.agents import stt
from livekit.agents.utils import AudioBuffer, merge_frames

from .config import SixtyDBConfig, http_session


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: str = "60db-stt",
        language: str = "en",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False, interim_results=False
            )
        )
        self._config = SixtyDBConfig.resolve(api_key=api_key, base_url=base_url)
        self._model = model
        self._language = language

    @staticmethod
    def _buffer_to_wav(buffer: AudioBuffer) -> bytes:
        """Collapse the captured frames into a single 16-bit PCM WAV blob."""
        frame = merge_frames(buffer)
        out = io.BytesIO()
        with wave.open(out, "wb") as wav:
            wav.setnchannels(frame.num_channels)
            wav.setsampwidth(2)  # 16-bit PCM
            wav.setframerate(frame.sample_rate)
            wav.writeframes(frame.data.tobytes())
        return out.getvalue()

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None = None,
        conn_options=None,
    ) -> stt.SpeechEvent:
        wav_bytes = self._buffer_to_wav(buffer)

        # 60dB API: multipart upload of a WAV clip -> JSON transcript.
        # Expected response shape: {"text": "...", "language": "en"}
        form = {
            "model": self._model,
            "language": language or self._language,
        }
        data = aiohttp_form(wav_bytes, form)

        async with http_session().post(
            self._config.url("/stt/recognize"),
            data=data,
            headers=self._config.auth_headers,
        ) as resp:
            resp.raise_for_status()
            payload = await resp.json()

        text = payload.get("text", "")
        detected_language = payload.get("language", language or self._language)

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(text=text, language=detected_language)
            ],
        )


def aiohttp_form(wav_bytes: bytes, fields: dict[str, str]):
    """Build a multipart form with the audio file plus scalar fields."""
    import aiohttp

    form = aiohttp.FormData()
    for key, value in fields.items():
        form.add_field(key, value)
    form.add_field(
        "file",
        wav_bytes,
        filename="audio.wav",
        content_type="audio/wav",
    )
    return form
