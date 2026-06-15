"""60dB text-to-speech for LiveKit Agents.

Implemented as a *non-streaming* (chunked) synthesizer: send the full
sentence, receive raw PCM audio, and emit it as LiveKit audio frames.
``VoicePipelineAgent`` wraps this with a sentence ``StreamAdapter`` so the
agent can still start speaking before the whole LLM response is ready.
"""

from __future__ import annotations

from livekit.agents import tts, utils

from .config import SixtyDBConfig, http_session

# 60dB API: we ask for raw little-endian 16-bit PCM at this rate so we can
# feed bytes straight into AudioByteStream without decoding a container.
SAMPLE_RATE = 24000
NUM_CHANNELS = 1


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = "andrew",
        model: str = "60db-tts",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._config = SixtyDBConfig.resolve(api_key=api_key, base_url=base_url)
        self._voice = voice
        self._model = model

    def synthesize(self, text: str, *, conn_options=None) -> "ChunkedStream":
        return ChunkedStream(
            tts=self,
            input_text=text,
            config=self._config,
            voice=self._voice,
            model=self._model,
        )


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: "TTS",
        input_text: str,
        config: SixtyDBConfig,
        voice: str,
        model: str,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text)
        self._config = config
        self._voice = voice
        self._model = model

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        bstream = utils.audio.AudioByteStream(
            sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS
        )

        # 60dB API: JSON in -> streamed raw PCM bytes out.
        body = {
            "model": self._model,
            "voice": self._voice,
            "text": self._input_text,
            "format": "pcm_s16le",
            "sample_rate": SAMPLE_RATE,
        }

        async with http_session().post(
            self._config.url("/tts/synthesize"),
            json=body,
            headers=self._config.auth_headers,
        ) as resp:
            resp.raise_for_status()
            async for chunk, _ in resp.content.iter_chunks():
                if not chunk:
                    continue
                for frame in bstream.write(chunk):
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(request_id=request_id, frame=frame)
                    )

        # Flush whatever partial frame remains in the buffer.
        for frame in bstream.flush():
            self._event_ch.send_nowait(
                tts.SynthesizedAudio(request_id=request_id, frame=frame)
            )
