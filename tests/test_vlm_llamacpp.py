"""Tests for the out-of-process llama-server scriptwriter (ticket #21).

Everything is injected — the process launcher, the health check, the
clock/sleep pair and the OpenAI-compatible client — so no binary, no
GGUF, no network and no ``openai`` package is needed, and the startup
wait loop runs on a fake clock.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import subprocess
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mangawhisperer.models import BoundingBox, SpeechBubble

pytest.importorskip("cv2")  # the shared image helpers live in vlm_api (cv2)

import cv2  # noqa: E402

from mangawhisperer.engines.factory import (  # noqa: E402
    ALL_PROVIDERS,
    create_reviewer,
    create_vlm_engine,
)
from mangawhisperer.engines.reviewer import LLMScriptReviewer  # noqa: E402
from mangawhisperer.engines.vlm_llamacpp import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_PORT,
    LlamaServerVisionLanguageEngine,
    build_panel_schema,
    default_health_check,
    model_label,
    resolve_server_url,
)

MODULE_LOGGER = "mangawhisperer.engines.vlm_llamacpp"
BBOX = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)
PANEL = np.full((200, 160, 3), 200, dtype=np.uint8)
BUBBLES = [
    SpeechBubble(text="Eu vou sobreviver!", bbox=BBOX),
    SpeechBubble(text="Griffith...!", bbox=BBOX),
]
SCRIPT_JSON = json.dumps(
    [
        {"text": "Eu vou sobreviver!", "speaker_id": "Guts", "is_speech": True},
        {"text": "Guts ergue a espada.", "speaker_id": "Narrator", "is_speech": False},
    ],
    ensure_ascii=False,
)
ENV_VARS = (
    "LLAMA_SERVER_BIN",
    "LLAMA_MODEL_GGUF",
    "LLAMA_MMPROJ_GGUF",
    "LLAMA_SERVER_URL",
    "LLAMA_SERVER_ARGS",
)


@pytest.fixture(autouse=True)
def _clean_llama_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# ── fakes ────────────────────────────────────────────────────────────


class FakeProcess:
    """Popen-like stand-in: ``poll``/``terminate``/``kill``/``wait``."""

    def __init__(self, exit_code: int | None = None, hang_on_terminate: bool = False) -> None:
        self.returncode = exit_code
        self.hang_on_terminate = hang_on_terminate
        self.terminated = False
        self.killed = False
        self.health_polls = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        if not self.hang_on_terminate:
            self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired(cmd="llama-server", timeout=timeout or 0)
        return self.returncode


class FakeServer:
    """A pretend llama-server tying the launcher and the health check.

    The health check reports ready once the *live* process has been
    polled ``ready_after`` times; with no live process nothing is ready
    (so a restart after ``release()`` has to spawn again). ``foreign``
    simulates some other server already answering on the port.
    """

    def __init__(
        self,
        ready_after: int | None = 2,
        exit_code: int | None = None,
        hang_on_terminate: bool = False,
        foreign: bool = False,
    ) -> None:
        self.ready_after = ready_after
        self.exit_code = exit_code
        self.hang_on_terminate = hang_on_terminate
        self.foreign = foreign
        self.launch_calls: list[list[str]] = []
        self.processes: list[FakeProcess] = []
        self.health_calls: list[str] = []

    def launch(self, args: list[str]) -> FakeProcess:
        self.launch_calls.append(list(args))
        proc = FakeProcess(self.exit_code, self.hang_on_terminate)
        self.processes.append(proc)
        return proc

    def health(self, url: str) -> bool:
        self.health_calls.append(url)
        if self.foreign:
            return True
        live = [p for p in self.processes if p.poll() is None]
        if not live or self.ready_after is None:
            return False
        live[-1].health_polls += 1
        return live[-1].health_polls >= self.ready_after


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeBadRequest(Exception):
    """What an OpenAI-compatible SDK raises for HTTP 400."""

    status_code = 400


class FakeCompletions:
    def __init__(self, content: str, fail_with: list[Exception] | None = None) -> None:
        self.content = content
        self.fail_with = list(fail_with or [])
        self.calls: list[dict] = []

    def create(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.fail_with:
            raise self.fail_with.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class FakeOpenAIClient:
    def __init__(self, content: str = SCRIPT_JSON, fail_with: list[Exception] | None = None) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(content, fail_with))

    @property
    def calls(self) -> list[dict]:
        return self.chat.completions.calls


def write_model_files(tmp_path: Path) -> SimpleNamespace:
    model = tmp_path / "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
    mmproj = tmp_path / "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"
    binary = tmp_path / "bin" / "llama-server.exe"
    binary.parent.mkdir(exist_ok=True)
    for path in (model, mmproj, binary):
        path.write_bytes(b"stub")
    return SimpleNamespace(model=model, mmproj=mmproj, binary=binary)


def write_gemma_files(tmp_path: Path, mmproj_name: str = "mmproj-gemma-4-E4B-it-Q8_0.gguf") -> SimpleNamespace:
    """A second model family in the same folder (the ADR's benchmark setup).

    ``mmproj_name`` defaults to ggml-org's naming; unsloth ships the same
    projector as the generic ``mmproj-F16.gguf``.
    """
    model = tmp_path / "gemma-4-E4B-it-Q4_K_M.gguf"
    mmproj = tmp_path / mmproj_name
    for path in (model, mmproj):
        path.write_bytes(b"stub")
    return SimpleNamespace(model=model, mmproj=mmproj)


def make_engine(tmp_path: Path, content: str = SCRIPT_JSON, **overrides):
    """Spawn-mode engine with every collaborator faked; returns (engine, world)."""
    files = write_model_files(tmp_path)
    server = overrides.pop("server", None) or FakeServer()
    client = overrides.pop("client", None) or FakeOpenAIClient(content)
    clock = FakeClock()
    kwargs = dict(
        model_path=files.model,
        mmproj_path=files.mmproj,
        server_binary=files.binary,
        launcher=server.launch,
        health_check=server.health,
        sleep=clock.sleep,
        clock=clock,
        client=client,
        startup_timeout_s=30,
    )
    kwargs.update(overrides)
    engine = LlamaServerVisionLanguageEngine(**kwargs)
    world = SimpleNamespace(server=server, client=client, clock=clock, files=files)
    return engine, world


# ── lifecycle (spawn mode) ───────────────────────────────────────────


class TestServerLifecycle:
    def test_construction_does_not_start_the_server(self, tmp_path) -> None:
        engine, world = make_engine(tmp_path)
        assert world.server.launch_calls == []
        assert world.server.health_calls == []
        assert engine.provider == "llamacpp"

    def test_first_contextualize_spawns_and_waits_for_health(self, tmp_path, caplog) -> None:
        engine, world = make_engine(tmp_path, server=FakeServer(ready_after=3))
        with caplog.at_level(logging.INFO, logger=MODULE_LOGGER):
            engine.contextualize(PANEL, BUBBLES)

        (args,) = world.server.launch_calls
        assert args[0] == str(world.files.binary)
        assert args[args.index("-m") + 1] == str(world.files.model)
        assert args[args.index("--mmproj") + 1] == str(world.files.mmproj)
        assert args[args.index("--host") + 1] == DEFAULT_HOST
        assert args[args.index("--port") + 1] == str(DEFAULT_PORT)
        assert args[args.index("-ngl") + 1] == "99"
        assert "-c" in args
        assert world.server.processes[0].health_polls == 3, "polled until /health said ok"
        assert all(url == f"http://{DEFAULT_HOST}:{DEFAULT_PORT}" for url in world.server.health_calls)
        assert any(
            "llama-server" in rec.getMessage() and "--mmproj" in rec.getMessage()
            for rec in caplog.records
            if rec.levelno == logging.INFO
        ), "the exact command line is logged at INFO"

    def test_running_server_is_reused_across_panels(self, tmp_path) -> None:
        engine, world = make_engine(tmp_path)
        engine.contextualize(PANEL, BUBBLES)
        engine.contextualize(PANEL, BUBBLES)
        assert len(world.server.launch_calls) == 1
        assert len(world.client.calls) == 2

    def test_startup_timeout_raises_ptbr_error_and_stops_the_process(self, tmp_path) -> None:
        engine, world = make_engine(tmp_path, server=FakeServer(ready_after=None))
        with pytest.raises(RuntimeError, match="não respondeu") as info:
            engine.contextualize(PANEL, BUBBLES)

        assert "30" in str(info.value), "the timeout is spelled out"
        assert world.clock.now >= 30 and world.clock.sleeps, "waited on the injected clock"
        assert world.server.processes[0].terminated, "no orphan server after a failed start"
        assert world.client.calls == []

    def test_server_exit_during_startup_raises(self, tmp_path) -> None:
        engine, _ = make_engine(tmp_path, server=FakeServer(exit_code=1))
        with pytest.raises(RuntimeError, match="encerrou") as info:
            engine.contextualize(PANEL, BUBBLES)
        assert "1" in str(info.value)

    def test_foreign_server_already_on_the_port_is_refused(self, tmp_path) -> None:
        engine, world = make_engine(tmp_path, server=FakeServer(foreign=True))
        with pytest.raises(RuntimeError, match="LLAMA_SERVER_URL"):
            engine.contextualize(PANEL, BUBBLES)
        assert world.server.launch_calls == [], "never spawns on top of another server"

    def test_release_terminates_is_idempotent_and_next_panel_restarts(self, tmp_path) -> None:
        engine, world = make_engine(tmp_path)
        engine.contextualize(PANEL, BUBBLES)
        first = world.server.processes[0]

        engine.release()
        assert first.terminated and first.poll() == 0
        engine.release()  # idempotent: nothing to do, no error

        engine.contextualize(PANEL, BUBBLES)
        assert len(world.server.launch_calls) == 2, "a later panel restarts the server"
        assert world.server.processes[1].poll() is None

    def test_release_kills_a_server_that_ignores_terminate(self, tmp_path) -> None:
        engine, world = make_engine(tmp_path, server=FakeServer(hang_on_terminate=True))
        engine.contextualize(PANEL, BUBBLES)
        engine.release()
        proc = world.server.processes[0]
        assert proc.terminated and proc.killed and proc.poll() is not None

    def test_crashed_server_is_restarted_on_the_next_panel(self, tmp_path, caplog) -> None:
        engine, world = make_engine(tmp_path)
        engine.contextualize(PANEL, BUBBLES)
        world.server.processes[0].returncode = 3  # died between panels

        with caplog.at_level(logging.WARNING, logger=MODULE_LOGGER):
            blocks = engine.contextualize(PANEL, BUBBLES)
        assert len(world.server.launch_calls) == 2
        assert blocks[0].speaker_id == "Guts"
        assert any("reinici" in rec.getMessage().lower() for rec in caplog.records)

    def test_port_none_picks_a_free_port(self, tmp_path) -> None:
        engine, world = make_engine(tmp_path, port=None)
        engine.contextualize(PANEL, BUBBLES)
        (args,) = world.server.launch_calls
        port = args[args.index("--port") + 1]
        assert port.isdigit() and int(port) > 0
        assert engine.server_url == f"http://{DEFAULT_HOST}:{port}"

    def test_extra_server_args_are_appended(self, tmp_path) -> None:
        engine, world = make_engine(tmp_path, extra_server_args=("--reasoning", "off"))
        engine.contextualize(PANEL, BUBBLES)
        (args,) = world.server.launch_calls
        assert args[-2:] == ["--reasoning", "off"]


# ── request / response ───────────────────────────────────────────────


class TestRequest:
    def test_payload_has_data_url_image_system_prompt_and_schema(self, tmp_path) -> None:
        engine, world = make_engine(tmp_path)
        engine.contextualize(PANEL, BUBBLES)

        (call,) = world.client.calls
        assert call["model"] == "Qwen3VL-8B-Instruct-Q4_K_M"
        system, user = call["messages"]
        assert system["role"] == "system" and "MangaWhisperer" in system["content"]
        assert "array JSON" in system["content"]
        image_part, text_part = user["content"]
        url = image_part["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert base64.standard_b64decode(url.split(",", 1)[1]).startswith(b"\x89PNG")
        assert "Eu vou sobreviver!" in text_part["text"] and "Griffith...!" in text_part["text"]

        response_format = call["response_format"]
        assert response_format["type"] == "json_schema"
        schema = response_format["json_schema"]["schema"]
        assert response_format["json_schema"]["name"]
        assert schema["type"] == "array"
        assert set(schema["items"]["required"]) == {"text", "speaker_id", "is_speech"}
        assert "sfx" not in schema["items"]["properties"], "no SFX tags -> no sfx field"

    def test_sfx_tags_become_a_closed_enum_in_the_schema(self, tmp_path) -> None:
        engine, world = make_engine(tmp_path, sfx_tags=("espada", "explosao"))
        engine.contextualize(PANEL, BUBBLES)
        schema = world.client.calls[0]["response_format"]["json_schema"]["schema"]
        assert schema["items"]["properties"]["sfx"]["enum"] == ["espada", "explosao"]
        assert "sfx" not in schema["items"]["required"]

    def test_parses_blocks_from_response(self, tmp_path) -> None:
        engine, _ = make_engine(tmp_path)
        blocks = engine.contextualize(PANEL, BUBBLES)
        assert [(b.text, b.speaker_id, b.is_speech) for b in blocks] == [
            ("Eu vou sobreviver!", "Guts", True),
            ("Guts ergue a espada.", "Narrator", False),
        ]

    def test_garbage_response_falls_back_to_passthrough(self, tmp_path) -> None:
        engine, _ = make_engine(tmp_path, content="sem json aqui")
        blocks = engine.contextualize(PANEL, BUBBLES)
        assert [(b.text, b.speaker_id) for b in blocks] == [
            ("Eu vou sobreviver!", "Desconhecido"),
            ("Griffith...!", "Desconhecido"),
        ]

    def test_request_failure_falls_back_to_passthrough_for_that_panel(self, tmp_path, caplog) -> None:
        client = FakeOpenAIClient(fail_with=[RuntimeError("connection reset")])
        engine, world = make_engine(tmp_path, client=client)
        with caplog.at_level(logging.ERROR, logger=MODULE_LOGGER):
            blocks = engine.contextualize(PANEL, BUBBLES)
        assert all(b.speaker_id == "Desconhecido" for b in blocks) and len(blocks) == 2
        assert any("connection reset" in rec.getMessage() for rec in caplog.records)

        assert engine.contextualize(PANEL, BUBBLES)[0].speaker_id == "Guts", "next panel is fine"
        assert len(world.server.launch_calls) == 1
        assert len(world.client.calls) == 2, "an error that is not a 400 gets no blind retry"

    def test_unrelated_bad_request_keeps_the_schema_for_later_panels(self, tmp_path, caplog) -> None:
        # llama-server also answers 400 for a prompt over the context size
        # (or an image sent to a text-only server, or an undecodable image):
        # a plain retry fails the same way, so the schema is not to blame.
        too_big = (
            "the request exceeds the available context size. try increasing the "
            "context size or enable context shift"
        )
        client = FakeOpenAIClient(fail_with=[FakeBadRequest(too_big), FakeBadRequest(too_big)])
        engine, world = make_engine(tmp_path, client=client)
        with caplog.at_level(logging.WARNING, logger=MODULE_LOGGER):
            blocks = engine.contextualize(PANEL, BUBBLES)
        assert all(b.speaker_id == "Desconhecido" for b in blocks), "that panel degrades"
        assert not any("rejeitou" in rec.getMessage() for rec in caplog.records), (
            "no misleading 'server rejected response_format' warning"
        )
        assert any("context size" in rec.getMessage() for rec in caplog.records), (
            "the real error reaches the log"
        )

        assert engine.contextualize(PANEL, BUBBLES)[0].speaker_id == "Guts"
        calls = world.client.calls
        assert len(calls) == 3, "schema attempt + plain retry on the bad panel, one call after"
        assert "response_format" in calls[0] and "response_format" not in calls[1]
        assert "response_format" in calls[2], "the next panel still asks for grammar JSON"
        assert engine.fingerprint.endswith("json=schema")

    def test_schema_rejected_by_server_falls_back_to_plain_json_instruction(self, tmp_path, caplog) -> None:
        client = FakeOpenAIClient(fail_with=[FakeBadRequest("response_format not supported")])
        engine, world = make_engine(tmp_path, client=client)
        with caplog.at_level(logging.WARNING, logger=MODULE_LOGGER):
            blocks = engine.contextualize(PANEL, BUBBLES)
        assert blocks[0].speaker_id == "Guts", "the retry without the schema succeeded"

        engine.contextualize(PANEL, BUBBLES)
        calls = world.client.calls
        assert len(calls) == 3, "one rejected + one retry, then a single call per panel"
        assert "response_format" in calls[0]
        assert "response_format" not in calls[1] and "response_format" not in calls[2]
        assert all("array JSON" in c["messages"][0]["content"] for c in calls)
        assert any("response_format" in rec.getMessage() for rec in caplog.records)

    def test_use_json_schema_false_sends_no_response_format(self, tmp_path) -> None:
        engine, world = make_engine(tmp_path, use_json_schema=False)
        engine.contextualize(PANEL, BUBBLES)
        assert "response_format" not in world.client.calls[0]

    def test_large_panels_are_downscaled_before_upload(self, tmp_path) -> None:
        engine, world = make_engine(tmp_path, max_image_edge=256)
        engine.contextualize(np.full((1000, 600, 3), 90, dtype=np.uint8), BUBBLES)
        url = world.client.calls[0]["messages"][1]["content"][0]["image_url"]["url"]
        png = base64.standard_b64decode(url.split(",", 1)[1])
        decoded = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        assert max(decoded.shape[:2]) == 256


# ── attach mode ──────────────────────────────────────────────────────


class TestAttachMode:
    def test_attach_never_spawns_and_release_only_warns(self, tmp_path, caplog) -> None:
        server = FakeServer(foreign=True)  # something is listening: that's the point
        engine, world = make_engine(tmp_path, server=server, server_url="http://127.0.0.1:9999/")
        assert engine.attached
        assert engine.server_url == "http://127.0.0.1:9999"

        blocks = engine.contextualize(PANEL, BUBBLES)
        assert blocks[0].speaker_id == "Guts"
        assert world.server.launch_calls == []

        with caplog.at_level(logging.WARNING, logger=MODULE_LOGGER):
            engine.release()
        assert any("VRAM" in rec.getMessage() for rec in caplog.records)
        assert world.server.launch_calls == []

    def test_preflight_requires_health(self, tmp_path) -> None:
        down, _ = make_engine(tmp_path, server=FakeServer(), server_url="http://127.0.0.1:9999")
        with pytest.raises(RuntimeError, match="127.0.0.1:9999"):
            down.preflight()

        up, _ = make_engine(tmp_path, server=FakeServer(foreign=True), server_url="http://127.0.0.1:9999")
        up.preflight()  # no error

    def test_preflight_and_first_panel_require_the_openai_sdk(self, monkeypatch) -> None:
        engine = LlamaServerVisionLanguageEngine(
            server_url="http://127.0.0.1:9999", health_check=FakeServer(foreign=True).health,
        )  # no injected client: the real SDK would be built on the first panel
        monkeypatch.setitem(sys.modules, "openai", None)  # a venv without the [vlm-api] extra
        with pytest.raises(RuntimeError, match="openai") as info:
            engine.preflight()
        assert "vlm-api" in str(info.value)

        with pytest.raises(RuntimeError, match="vlm-api"):
            engine.contextualize(PANEL, BUBBLES)  # fatal, never a silent passthrough volume

    def test_model_label_is_sent_verbatim_for_ollama_style_servers(self, tmp_path) -> None:
        engine, world = make_engine(
            tmp_path, server=FakeServer(foreign=True), server_url="http://localhost:11434/v1",
            model_path="qwen3-vl:8b", mmproj_path=None, server_binary=None,
        )
        assert engine.model == "qwen3-vl:8b"
        engine.contextualize(PANEL, BUBBLES)
        assert world.client.calls[0]["model"] == "qwen3-vl:8b"


# ── preflight (spawn mode) ───────────────────────────────────────────


class TestPreflightSpawnMode:
    def test_missing_binary_says_how_to_install_and_which_env_var(self, tmp_path) -> None:
        engine, _ = make_engine(tmp_path, server_binary=tmp_path / "nope" / "llama-server.exe")
        with pytest.raises(RuntimeError, match="LLAMA_SERVER_BIN") as info:
            engine.preflight()
        assert "llama.cpp" in str(info.value) and "releases" in str(info.value)

    def test_missing_model_names_env_var(self, tmp_path) -> None:
        engine, _ = make_engine(tmp_path, model_path=tmp_path / "missing.gguf")
        with pytest.raises(RuntimeError, match="LLAMA_MODEL_GGUF"):
            engine.preflight()

        unset, _ = make_engine(tmp_path, model_path=None)
        with pytest.raises(RuntimeError, match="LLAMA_MODEL_GGUF"):
            unset.preflight()

    def test_missing_mmproj_names_env_var(self, tmp_path) -> None:
        engine, _ = make_engine(tmp_path, mmproj_path=tmp_path / "missing-mmproj.gguf")
        with pytest.raises(RuntimeError, match="LLAMA_MMPROJ_GGUF"):
            engine.preflight()

    def test_mmproj_sibling_is_auto_discovered(self, tmp_path, caplog) -> None:
        engine, world = make_engine(tmp_path, mmproj_path=None)
        with caplog.at_level(logging.INFO, logger=MODULE_LOGGER):
            engine.contextualize(PANEL, BUBBLES)
        (args,) = world.server.launch_calls
        assert args[args.index("--mmproj") + 1] == str(world.files.mmproj)

    def test_no_mmproj_anywhere_warns_but_starts(self, tmp_path, caplog) -> None:
        engine, world = make_engine(tmp_path, mmproj_path=None)
        world.files.mmproj.unlink()
        with caplog.at_level(logging.WARNING, logger=MODULE_LOGGER):
            engine.contextualize(PANEL, BUBBLES)
        (args,) = world.server.launch_calls
        assert "--mmproj" not in args
        assert any("mmproj" in rec.getMessage() for rec in caplog.records)

    def test_two_model_families_in_one_folder_each_get_their_own_mmproj(self, tmp_path) -> None:
        gemma = write_gemma_files(tmp_path)
        qwen_engine, qwen_world = make_engine(tmp_path, mmproj_path=None)
        qwen_engine.contextualize(PANEL, BUBBLES)
        (args,) = qwen_world.server.launch_calls
        assert args[args.index("--mmproj") + 1] == str(qwen_world.files.mmproj)

        gemma_engine, gemma_world = make_engine(tmp_path, model_path=gemma.model, mmproj_path=None)
        gemma_engine.contextualize(PANEL, BUBBLES)
        (args,) = gemma_world.server.launch_calls
        assert args[args.index("--mmproj") + 1] == str(gemma.mmproj)

    def test_mmproj_of_another_family_is_never_borrowed(self, tmp_path) -> None:
        # unsloth's Gemma projector is the generic mmproj-F16.gguf: nothing in
        # its name says "gemma", while Qwen's projector sits right beside it.
        gemma = write_gemma_files(tmp_path, mmproj_name="mmproj-F16.gguf")
        engine, world = make_engine(tmp_path, model_path=gemma.model, mmproj_path=None)
        with pytest.raises(RuntimeError, match="LLAMA_MMPROJ_GGUF") as info:
            engine.contextualize(PANEL, BUBBLES)
        assert "mmproj-F16.gguf" in str(info.value) and world.files.mmproj.name in str(info.value), (
            "the error lists the candidates the user must choose from"
        )
        assert world.server.launch_calls == [], "no server with a projector that may be another model's"

        qwen_engine, qwen_world = make_engine(tmp_path, mmproj_path=None)
        qwen_engine.preflight()
        assert qwen_engine.mmproj_path == qwen_world.files.mmproj, "Qwen still finds its own projector"

        explicit, _ = make_engine(tmp_path, model_path=gemma.model, mmproj_path=gemma.mmproj)
        explicit.preflight()  # LLAMA_MMPROJ_GGUF settles it
        assert explicit.mmproj_path == gemma.mmproj

    def test_lone_mmproj_is_used_even_without_the_model_name(self, tmp_path) -> None:
        gemma = write_gemma_files(tmp_path, mmproj_name="mmproj-F16.gguf")
        engine, world = make_engine(tmp_path, model_path=gemma.model, mmproj_path=None)
        world.files.mmproj.unlink()  # only Gemma's generic projector remains
        engine.preflight()
        assert engine.mmproj_path == gemma.mmproj

    def test_same_projector_in_two_quantizations_must_be_chosen_explicitly(self, tmp_path) -> None:
        (tmp_path / "mmproj-Qwen3VL-8B-Instruct-F16.gguf").write_bytes(b"stub")
        engine, _ = make_engine(tmp_path, mmproj_path=None)
        with pytest.raises(RuntimeError, match="LLAMA_MMPROJ_GGUF") as info:
            engine.preflight()
        assert "Q8_0" in str(info.value) and "F16" in str(info.value)

    def test_closest_name_wins_between_sizes_of_one_family(self, tmp_path) -> None:
        # Qwen3-VL 4B and 8B downloaded side by side: both projectors carry
        # "qwen3vl", only one carries "8b" + "instruct" like the model.
        small = tmp_path / "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"
        small.write_bytes(b"stub")
        (tmp_path / "Qwen3VL-4B-Instruct-Q4_K_M.gguf").write_bytes(b"stub")
        engine, world = make_engine(tmp_path, mmproj_path=None)
        engine.preflight()
        assert engine.mmproj_path == world.files.mmproj

    def test_preflight_requires_the_openai_sdk_before_spawning(self, tmp_path, monkeypatch) -> None:
        files = write_model_files(tmp_path)
        server = FakeServer()
        engine = LlamaServerVisionLanguageEngine(
            model_path=files.model, mmproj_path=files.mmproj, server_binary=files.binary,
            launcher=server.launch, health_check=server.health,
        )  # no injected client: the real SDK would be built on the first panel
        monkeypatch.setitem(sys.modules, "openai", None)  # a venv without the [vlm-api] extra
        with pytest.raises(RuntimeError, match="openai") as info:
            engine.preflight()
        assert "vlm-api" in str(info.value)

        with pytest.raises(RuntimeError, match="vlm-api"):
            engine.contextualize(PANEL, BUBBLES)
        assert server.launch_calls == [], "the 5 GB server is never launched for nothing"

    def test_injected_client_needs_no_openai_sdk(self, tmp_path, monkeypatch) -> None:
        engine, _ = make_engine(tmp_path)
        monkeypatch.setitem(sys.modules, "openai", None)
        engine.preflight()
        assert engine.contextualize(PANEL, BUBBLES)[0].speaker_id == "Guts"

    def test_binary_resolves_from_env_then_path(self, tmp_path, monkeypatch) -> None:
        files = write_model_files(tmp_path)
        monkeypatch.setenv("LLAMA_SERVER_BIN", str(files.binary))
        engine, _ = make_engine(tmp_path, server_binary=None)
        engine.preflight()
        assert engine.server_binary == files.binary

        monkeypatch.delenv("LLAMA_SERVER_BIN")
        monkeypatch.setattr("shutil.which", lambda name: str(files.binary) if name == "llama-server" else None)
        engine, _ = make_engine(tmp_path, server_binary=None)
        engine.preflight()
        assert engine.server_binary == files.binary

    def test_ensure_server_runs_preflight(self, tmp_path) -> None:
        engine, world = make_engine(tmp_path, model_path=tmp_path / "missing.gguf")
        with pytest.raises(RuntimeError, match="LLAMA_MODEL_GGUF"):
            engine.contextualize(PANEL, BUBBLES)
        assert world.server.launch_calls == []


# ── fingerprint ──────────────────────────────────────────────────────


class TestFingerprint:
    def test_includes_gguf_stem_and_prompt_digest(self, tmp_path) -> None:
        engine, _ = make_engine(tmp_path)
        digest = hashlib.sha1(engine.system_prompt.encode("utf-8")).hexdigest()[:8]
        assert engine.fingerprint.startswith("vlm-llamacpp:Qwen3VL-8B-Instruct-Q4_K_M:")
        assert f"prompt={digest}" in engine.fingerprint

    def test_prompt_changes_invalidate_checkpoints(self, tmp_path) -> None:
        plain, _ = make_engine(tmp_path)
        with_sfx, _ = make_engine(tmp_path, sfx_tags=("espada",))
        other_cast, _ = make_engine(tmp_path, known_characters=("Ippo", "Takamura"))
        assert len({plain.fingerprint, with_sfx.fingerprint, other_cast.fingerprint}) == 3


# ── helpers ──────────────────────────────────────────────────────────


class TestHelpers:
    def test_default_health_check_maps_http_statuses(self, monkeypatch) -> None:
        outcomes: dict[str, object] = {}

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(url, timeout):
            outcomes["url"] = url
            result = outcomes["result"]
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        base = "http://127.0.0.1:8080"

        outcomes["result"] = Response()
        assert default_health_check(base) is True
        assert outcomes["url"] == base + "/health"

        outcomes["result"] = urllib.error.HTTPError(base, 503, "Loading model", {}, None)
        assert default_health_check(base) is False

        outcomes["result"] = urllib.error.HTTPError(base, 404, "not found", {}, None)
        assert default_health_check(base) is True, "servers without /health (Ollama) are up"

        outcomes["result"] = urllib.error.URLError("connection refused")
        assert default_health_check(base) is False

    def test_resolve_server_url_normalizes_and_defaults(self) -> None:
        assert resolve_server_url({}) == f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
        assert resolve_server_url({"LLAMA_SERVER_URL": "http://localhost:11434/v1/"}) == "http://localhost:11434"
        assert resolve_server_url({"LLAMA_SERVER_URL": ""}) == f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"

    def test_model_label(self) -> None:
        assert model_label("C:/models/Qwen3VL-8B-Instruct-Q4_K_M.gguf") == "Qwen3VL-8B-Instruct-Q4_K_M"
        assert model_label(Path("gemma-4-E4B-it-Q4_K_M.GGUF")) == "gemma-4-E4B-it-Q4_K_M"
        assert model_label("qwen3-vl:8b") == "qwen3-vl:8b"
        assert model_label(None) == "llama-server"

    def test_panel_schema_is_the_block_contract(self) -> None:
        schema = build_panel_schema(("espada",))
        item = schema["items"]
        assert item["properties"]["text"]["type"] == "string"
        assert item["properties"]["is_speech"]["type"] == "boolean"
        assert item["properties"]["sfx"]["enum"] == ["espada"]
        assert item["additionalProperties"] is False


# ── factory ──────────────────────────────────────────────────────────


class TestFactory:
    def test_llamacpp_is_a_registered_provider(self) -> None:
        assert "llamacpp" in ALL_PROVIDERS

    def test_reads_env_vars(self, tmp_path, monkeypatch) -> None:
        files = write_model_files(tmp_path)
        monkeypatch.setenv("LLAMA_SERVER_BIN", str(files.binary))
        monkeypatch.setenv("LLAMA_MODEL_GGUF", str(files.model))
        monkeypatch.setenv("LLAMA_MMPROJ_GGUF", str(files.mmproj))
        monkeypatch.setenv("LLAMA_SERVER_ARGS", "--reasoning off --image-max-tokens 1024")

        engine = create_vlm_engine("llamacpp", sfx_tags=("espada",))
        assert isinstance(engine, LlamaServerVisionLanguageEngine)
        assert engine.provider == "llamacpp"
        assert engine.model == "Qwen3VL-8B-Instruct-Q4_K_M"
        assert engine.model_path == files.model
        assert engine.mmproj_path == files.mmproj
        assert engine.server_binary == files.binary
        assert engine.extra_server_args == ("--reasoning", "off", "--image-max-tokens", "1024")
        assert not engine.attached
        engine.preflight()  # everything exists

    def test_model_override_is_the_gguf_path(self, tmp_path, monkeypatch) -> None:
        files = write_model_files(tmp_path)
        monkeypatch.setenv("LLAMA_MODEL_GGUF", str(tmp_path / "other.gguf"))
        engine = create_vlm_engine("llamacpp", model=str(files.model))
        assert engine.model_path == files.model

    def test_without_any_configuration_preflight_lists_every_gap(self, monkeypatch) -> None:
        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setitem(sys.modules, "openai", None)
        engine = create_vlm_engine("llamacpp")
        assert engine.model_path is None and not engine.attached
        with pytest.raises(RuntimeError) as info:
            engine.preflight()
        message = str(info.value)
        assert "LLAMA_MODEL_GGUF" in message and "LLAMA_SERVER_BIN" in message, (
            "one error names everything missing, not just the first gap"
        )
        assert "vlm-api" in message, "the missing SDK is one of the listed gaps"

    def test_server_url_env_selects_attach_mode(self, monkeypatch) -> None:
        monkeypatch.setenv("LLAMA_SERVER_URL", "http://localhost:11434/v1")
        engine = create_vlm_engine("llamacpp", model="qwen3-vl:8b")
        assert engine.attached
        assert engine.server_url == "http://localhost:11434"
        assert engine.model == "qwen3-vl:8b"

    def test_existing_providers_unchanged(self) -> None:
        from mangawhisperer.engines.placeholders import PassthroughVLM
        from mangawhisperer.engines.vlm_api import OpenAICompatibleVisionLanguageEngine

        assert isinstance(create_vlm_engine("passthrough"), PassthroughVLM)
        assert isinstance(create_vlm_engine("qwen"), OpenAICompatibleVisionLanguageEngine)
        assert create_reviewer("qwen-local") is None
        with pytest.raises(ValueError):
            create_vlm_engine("gemini")

    def test_reviewer_points_at_the_same_server(self, tmp_path, monkeypatch) -> None:
        files = write_model_files(tmp_path)
        reviewer = create_reviewer("llamacpp", model=str(files.model), sfx_tags=("espada",))
        assert isinstance(reviewer, LLMScriptReviewer)
        assert reviewer.provider == "llamacpp"
        assert reviewer.model == "Qwen3VL-8B-Instruct-Q4_K_M"
        assert reviewer.base_url == f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/v1"
        assert reviewer.fingerprint.startswith("reviewer:llamacpp:Qwen3VL-8B-Instruct-Q4_K_M:")

        monkeypatch.setenv("LLAMA_SERVER_URL", "http://localhost:11434/v1")
        attached = create_reviewer("llamacpp", model="qwen3-vl:8b")
        assert attached.base_url == "http://localhost:11434/v1"
        assert attached.model == "qwen3-vl:8b"

    def test_reviewer_llamacpp_reviews_through_the_openai_compatible_path(self) -> None:
        from mangawhisperer.models import ContextualizedBlock, PanelData

        payload = json.dumps({
            "panels": [{"panel": 0, "blocks": [
                {"text": "Você é meu!", "speaker_id": "Criatura", "is_speech": True}
            ]}]
        }, ensure_ascii=False)
        client = FakeOpenAIClient(content=payload)
        reviewer = LLMScriptReviewer(provider="llamacpp", model="local", client=client,
                                     base_url="http://127.0.0.1:8080/v1")
        panel = PanelData(
            image_path=Path("p0.npy"), bbox=BBOX, page_number=1, panel_index=0,
            blocks=[ContextualizedBlock(text="Você é meu!", speaker_id="Monstro", is_speech=True)],
        )
        result = reviewer.review([panel])
        assert result[0].blocks[0].speaker_id == "Criatura"
        assert client.calls[0]["model"] == "local"

    def test_reviewer_llamacpp_builds_a_local_client_without_api_key(self, monkeypatch) -> None:
        openai = pytest.importorskip("openai")
        captured: dict = {}

        def fake_openai(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(kind="fake")

        monkeypatch.setattr(openai, "OpenAI", fake_openai)
        reviewer = LLMScriptReviewer(provider="llamacpp", base_url="http://127.0.0.1:8080/v1")
        assert reviewer._get_openai_client().kind == "fake"
        assert captured["base_url"] == "http://127.0.0.1:8080/v1"
        assert captured["api_key"], "the SDK needs a non-empty placeholder key"
