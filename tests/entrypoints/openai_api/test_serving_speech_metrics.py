# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Prometheus metric coverage for the Speech API audio path."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.entrypoints.openai import serving_speech as speech_module
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _MetricsStub:
    def __init__(self) -> None:
        self.ttfp_calls: list[tuple[str, str, float]] = []
        self.underrun_calls: list[tuple[str, str, float]] = []
        self.continuity_calls: list[tuple[str, str, int]] = []

    def observe_audio_ttfp(self, stage: str, replica: str, seconds: float) -> None:
        self.ttfp_calls.append((stage, replica, seconds))

    def observe_audio_underrun(self, stage: str, replica: str, seconds: float) -> None:
        self.underrun_calls.append((stage, replica, seconds))

    def inc_audio_continuity_ok(self, stage: str, replica: str, threshold_ms: int) -> None:
        self.continuity_calls.append((stage, replica, threshold_ms))


def _serving(metrics: _MetricsStub, *, adapter=None) -> OmniOpenAIServingSpeech:
    serving = OmniOpenAIServingSpeech.__new__(OmniOpenAIServingSpeech)
    serving._tts_model_type = "qwen3_tts"
    serving.engine_client = SimpleNamespace(mod_metrics=metrics, request_states={})
    serving._get_tts_adapter = lambda: adapter
    serving.create_audio = lambda audio_obj: SimpleNamespace(
        audio_data=b"\0\0" * int(audio_obj.audio_tensor.size),
        media_type="audio/pcm",
    )
    serving._mark_ref_audio_artifact_ready_for_request = lambda request_id: None
    serving._discard_ref_audio_artifact_warmup = lambda request_id: None
    return serving


def _result(samples: int = 320, *, stage_id: int = 1, replica_id: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        multimodal_output={"audio": torch.zeros(samples), "sr": 16000},
        stage_id=stage_id,
        replica_id=replica_id,
    )


async def _generate(*results):
    for result in results:
        yield result


@pytest.mark.asyncio
async def test_streaming_speech_observes_ttfp_once_on_first_pcm_payload(monkeypatch):
    metrics = _MetricsStub()
    serving = _serving(metrics)
    monkeypatch.setattr("vllm_omni.entrypoints.openai.serving_speech.time.time", lambda: 100.25)

    chunks = serving._generate_audio_chunks(
        _generate(_result(), _result()),
        request_id="speech-test",
        request_arrival_ts=100.0,
    )
    assert len([chunk async for chunk in chunks]) == 2
    assert metrics.ttfp_calls == [("1", "2", pytest.approx(0.25))]
    assert metrics.underrun_calls == [("1", "2", pytest.approx(0.0))]
    assert metrics.continuity_calls == [("1", "2", 100)]


@pytest.mark.asyncio
async def test_streaming_speech_does_not_count_empty_payload_as_first_packet(monkeypatch):
    metrics = _MetricsStub()
    serving = _serving(metrics)
    monkeypatch.setattr("vllm_omni.entrypoints.openai.serving_speech.time.time", lambda: 100.25)

    chunks = serving._generate_audio_chunks(
        _generate(_result(samples=0), _result()),
        request_id="speech-test",
        request_arrival_ts=100.0,
    )
    assert len([chunk async for chunk in chunks]) == 2
    assert metrics.ttfp_calls == [("1", "2", pytest.approx(0.25))]
    assert metrics.underrun_calls == [("1", "2", pytest.approx(0.0))]
    assert metrics.continuity_calls == [("1", "2", 100)]


@pytest.mark.asyncio
async def test_streaming_speech_does_not_finalize_continuity_on_error(monkeypatch):
    metrics = _MetricsStub()
    serving = _serving(metrics)
    monkeypatch.setattr("vllm_omni.entrypoints.openai.serving_speech.time.time", lambda: 100.25)

    async def failing_stream():
        yield _result()
        raise RuntimeError("stream failed")

    chunks = serving._generate_audio_chunks(
        failing_stream(),
        request_id="speech-test",
        request_arrival_ts=100.0,
    )
    with pytest.raises(RuntimeError, match="stream failed"):
        _ = [chunk async for chunk in chunks]

    assert len(metrics.ttfp_calls) == 1
    assert metrics.underrun_calls == []
    assert metrics.continuity_calls == []


@pytest.mark.asyncio
async def test_streaming_speech_does_not_finalize_continuity_on_validation_error(monkeypatch):
    metrics = _MetricsStub()

    def reject_generation(_tts_params, **_kwargs):
        raise RuntimeError("generation validation failed")

    adapter = SimpleNamespace(validates_generation=True, validate_generation=reject_generation)
    serving = _serving(metrics, adapter=adapter)
    monkeypatch.setattr("vllm_omni.entrypoints.openai.serving_speech.time.time", lambda: 100.25)

    chunks = serving._generate_audio_chunks(
        _generate(_result()),
        request_id="speech-test",
        request_arrival_ts=100.0,
        tts_params={"task_type": ["Base"]},
    )
    with pytest.raises(RuntimeError, match="generation validation failed"):
        _ = [chunk async for chunk in chunks]

    assert len(metrics.ttfp_calls) == 1
    assert metrics.underrun_calls == []
    assert metrics.continuity_calls == []


@pytest.mark.asyncio
async def test_streaming_speech_reports_late_chunk_as_underrun(monkeypatch):
    metrics = _MetricsStub()
    serving = _serving(metrics)
    monkeypatch.setattr("vllm_omni.entrypoints.openai.serving_speech.time.time", lambda: 100.25)
    perf_times = iter((0.0, 0.1, 0.1, 2.0, 2.0))
    monkeypatch.setattr("vllm_omni.entrypoints.openai.serving_speech.time.perf_counter", lambda: next(perf_times))

    chunks = serving._generate_audio_chunks(
        _generate(_result(), _result()),
        request_id="speech-test",
        request_arrival_ts=100.0,
    )
    assert len([chunk async for chunk in chunks]) == 2
    assert metrics.underrun_calls[0][:2] == ("1", "2")
    assert metrics.underrun_calls[0][2] > 0.1
    assert metrics.continuity_calls == []


@pytest.mark.asyncio
async def test_non_streaming_speech_does_not_observe_ttfp():
    metrics = _MetricsStub()
    serving = _serving(metrics)
    serving._audio_encode_speed = lambda _request: 1.0

    async def prepare(_request, **_kwargs):
        result = _result()
        result.metrics = {}
        return "speech-test", _generate(result), {}

    serving._prepare_speech_generation = prepare
    request = OpenAICreateSpeechRequest(input="hello", response_format="pcm")

    audio_data, media_type = await serving._generate_audio_bytes(request, request_arrival_ts=100.0)

    assert audio_data
    assert media_type == "audio/pcm"
    assert metrics.ttfp_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("flush_only", [False, True])
@pytest.mark.parametrize("response_format", ["pcm", "wav"])
async def test_resampled_speech_records_flush_with_audio_producer(monkeypatch, flush_only, response_format):
    metrics = _MetricsStub()
    serving = _serving(metrics)
    clock = {"now": 0.0}
    finalized = []
    original_finalize = speech_module.observe_audio_streaming_finalize

    def capture_finalize(*args, **kwargs):
        finalized.append(kwargs)
        original_finalize(*args, **kwargs)

    class BufferedResampler:
        def __init__(self, source_rate, target_rate):
            assert (source_rate, target_rate) == (16000, 24000)

        def process(self, chunk, *, final=False):
            clock["now"] = 0.5 if final else 0.25
            if not final and flush_only:
                return np.empty(0, dtype=np.float32)
            return np.zeros(2400, dtype=np.float32)

    monkeypatch.setattr(speech_module, "StreamingAudioResampler", BufferedResampler)
    monkeypatch.setattr(speech_module, "observe_audio_streaming_finalize", capture_finalize)
    monkeypatch.setattr(speech_module.time, "time", lambda: 100.0 + clock["now"])
    monkeypatch.setattr(speech_module.time, "perf_counter", lambda: clock["now"])
    # The last result is not audio and must not supply the flush metric labels.
    non_audio = SimpleNamespace(multimodal_output={"timestamps": []}, stage_id=9, replica_id=8)
    chunks = [
        chunk
        async for chunk in serving._generate_audio_chunks(
            _generate(_result(), non_audio),
            request_id="speech-test",
            response_format=response_format,
            request_start_s=0.0,
            request_arrival_ts=100.0,
            target_sample_rate=24000,
        )
    ]

    if response_format == "wav":
        assert chunks.pop(0).startswith(b"RIFF")
    expected_arrivals = [0.5] if flush_only else [0.25, 0.5]
    assert [len(chunk) for chunk in chunks] == [4800] * len(expected_arrivals)
    assert metrics.ttfp_calls == [("1", "2", pytest.approx(expected_arrivals[0]))]
    assert len(finalized) == 1
    assert finalized[0]["sample_rate"] == 24000
    assert finalized[0]["channels"] == 1
    assert finalized[0]["chunk_bytes"] == [4800] * len(expected_arrivals)
    assert finalized[0]["chunk_arrival_times_s"] == expected_arrivals
    assert metrics.underrun_calls == [("1", "2", pytest.approx(0.0 if flush_only else 0.15))]
    assert metrics.continuity_calls == ([("1", "2", 100)] if flush_only else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_pcm", [False, True])
async def test_resampled_speech_without_pcm_does_not_emit_metrics(monkeypatch, empty_pcm):
    metrics = _MetricsStub()
    serving = _serving(metrics)

    class EmptyResampler:
        def __init__(self, source_rate, target_rate):
            pass

        def process(self, chunk, *, final=False):
            # Also cover a nonempty flush waveform whose encoder emits no PCM.
            return np.zeros(10 if final and empty_pcm else 0, dtype=np.float32)

    monkeypatch.setattr(speech_module, "StreamingAudioResampler", EmptyResampler)
    serving.create_audio = lambda audio_obj: SimpleNamespace(audio_data=b"", media_type="audio/pcm")
    chunks = [
        chunk
        async for chunk in serving._generate_audio_chunks(
            _generate(_result()),
            request_id="speech-test",
            request_arrival_ts=100.0,
            target_sample_rate=24000,
        )
    ]
    assert not any(chunks)
    assert metrics.ttfp_calls == []
    assert metrics.underrun_calls == []
    assert metrics.continuity_calls == []
