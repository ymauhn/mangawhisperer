"""Out-of-process local scriptwriter: llama.cpp's ``llama-server`` (ADR-0004).

The engine owns the server's lifecycle. On the first panel it spawns
``llama-server`` with the GGUF weights and the multimodal projector,
waits for ``/health`` to report the model loaded, and from then on talks
to it through the OpenAI-compatible ``/v1/chat/completions`` endpoint —
JSON output constrained by a grammar built from the block schema
(``response_format`` of type ``json_schema``). ``release()`` terminates
the process so the TTS model can have the 8 GB GPU to itself; the next
panel restarts it. *Attach mode* (``server_url``) reuses a server
someone else started — llama-server, Ollama's OpenAI endpoint — and
never spawns or kills anything.

Every collaborator (process launcher, health probe, clock/sleep and the
HTTP client) is injectable, so the tests run without the binary, a
model or a network. The OpenAI SDK is imported lazily — and its
presence is checked in ``preflight()``, before any server is spawned.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import weakref
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from mangawhisperer.engines.script_parsing import parse_script_blocks, passthrough_blocks
from mangawhisperer.engines.vlm import DEFAULT_CAST, build_scriptwriter_prompt
from mangawhisperer.engines.vlm_api import JSON_INSTRUCTION, bubble_request_text, png_data_url
from mangawhisperer.interfaces import Image, VisionLanguageEngine
from mangawhisperer.models import VOICE_PROFILES, ContextualizedBlock, SpeechBubble

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
"""llama-server's own default; fixed so the reviewer can find the same server."""
DEFAULT_BINARY = "llama-server"
DEFAULT_CTX_SIZE = 4096
"""KV budget: ~0.6 GB for an 8B model at 4k tokens (research R1, §4) — the
slack an 8 GB GPU has left after Q4_K_M weights + mmproj."""
DEFAULT_MAX_IMAGE_EDGE = 1024
"""Long-edge cap before upload: ~768 visual tokens for Qwen3-VL at 32 px/token."""

ENV_SERVER_BIN = "LLAMA_SERVER_BIN"
ENV_MODEL_GGUF = "LLAMA_MODEL_GGUF"
ENV_MMPROJ_GGUF = "LLAMA_MMPROJ_GGUF"
ENV_SERVER_URL = "LLAMA_SERVER_URL"

_HEALTH_POLL_INTERVAL_S = 0.5
_TERMINATE_GRACE_S = 10.0
_INSTALL_HINT = (
    "Instale o llama.cpp — no Windows: baixe llama-<tag>-bin-win-cuda-13.3-x64.zip "
    "(+ cudart-llama-bin-win-cuda-13.3-x64.zip, se não houver CUDA no PATH) em "
    "https://github.com/ggml-org/llama.cpp/releases, ou `winget install llama.cpp` "
    "(build Vulkan), ou `scoop install versions/llama.cpp-cu124` — e defina "
    f"{ENV_SERVER_BIN}=<pasta>\\llama-server.exe (ou coloque a pasta no PATH)."
)
_OPENAI_HINT = (
    "O provedor llamacpp fala com o servidor pelo SDK da OpenAI, que não está instalado: "
    'execute `pip install -e ".[vlm-api]"` (ou `pip install openai`).'
)

Launcher = Callable[[list[str]], Any]
"""Takes the argv list, returns a Popen-like object (poll/terminate/kill/wait)."""
HealthCheck = Callable[[str], bool]
"""Takes the server base URL, returns whether it is up and the model loaded."""


# ── module-level helpers (shared with the factory / reviewer) ───────


def model_label(model: str | os.PathLike[str] | None) -> str:
    """The name reported as ``model``: a GGUF's stem, or the string as
    given (an Ollama tag such as ``qwen3-vl:8b``); ``llama-server`` when
    nothing is configured (attach mode to a single-model server)."""
    if not model:
        return "llama-server"
    text = str(model)
    return Path(text).stem if text.lower().endswith(".gguf") else text


def resolve_server_url(
    env: Mapping[str, str], host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> str:
    """Base URL of the server both the scriptwriter and the reviewer use:
    ``LLAMA_SERVER_URL`` (attach mode) or the address a spawned server
    listens on. Trailing slashes and a ``/v1`` suffix are dropped so the
    OpenAI base URL of an Ollama install works verbatim."""
    raw = (env.get(ENV_SERVER_URL) or "").strip()
    return _normalize_url(raw) if raw else f"http://{host}:{port}"


def build_panel_schema(sfx_tags: Sequence[str] = ()) -> dict[str, Any]:
    """JSON Schema of the scriptwriter's answer (the array the shared
    parser expects). ``sfx`` only exists when a library is configured,
    and then as a closed enum so the grammar cannot invent tags."""
    properties: dict[str, Any] = {
        "text": {"type": "string"},
        "speaker_id": {"type": "string"},
        "is_speech": {"type": "boolean"},
    }
    if sfx_tags:
        properties["sfx"] = {"type": "string", "enum": list(sfx_tags)}
    properties["voice"] = {"type": "string", "enum": list(VOICE_PROFILES)}
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": properties,
            "required": ["text", "speaker_id", "is_speech"],
            "additionalProperties": False,
        },
    }


def default_health_check(url: str, timeout_s: float = 2.0) -> bool:
    """``GET <url>/health``: 200 = model loaded; 503 = still loading;
    404 = a server without that route (Ollama) that is nevertheless up;
    anything else (connection refused, timeout) = not ready."""
    import urllib.error  # noqa: PLC0415 — keep the module import light
    import urllib.request  # noqa: PLC0415

    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout_s) as response:
            return 200 <= response.status < 300
    except urllib.error.HTTPError as exc:
        return exc.code == 404
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _normalize_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if url.lower().endswith("/v1"):
        url = url[:-3]
    return url.rstrip("/")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((DEFAULT_HOST, 0))
        return int(sock.getsockname()[1])


def _terminate_process(proc: Any) -> None:
    """Stop a server process: terminate, then kill if it lingers.

    Module-level (not a method) so a ``weakref.finalize`` can run it at
    interpreter exit without keeping the engine alive.
    """
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except OSError:
        pass
    try:
        proc.wait(timeout=_TERMINATE_GRACE_S)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        proc.kill()
        proc.wait(timeout=_TERMINATE_GRACE_S)
    except (OSError, subprocess.TimeoutExpired):
        logger.warning("llama-server não encerrou após kill(); a VRAM pode continuar ocupada")


def _looks_like_bad_request(exc: BaseException) -> bool:
    """Whether the server refused the *request shape* (HTTP 400/422, or
    an error naming the JSON-constraint fields), as opposed to failing
    while serving it.

    Grounds to retry without ``response_format`` — not proof the schema
    was the problem: llama-server also answers 400 for a prompt over
    the context size, an image sent to a text-only server or an
    undecodable image. Only a plain retry that succeeds settles it.
    """
    if getattr(exc, "status_code", None) in (400, 422):
        return True
    name = type(exc).__name__.lower()
    if "badrequest" in name or "unprocessable" in name:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in ("response_format", "json_schema", "grammar"))


def _name_tokens(name: str) -> list[str]:
    """Lower-case alphanumeric runs of a file stem: ``Qwen3VL-8B-Instruct-Q4_K_M``
    -> ``qwen3vl, 8b, instruct, q4, k, m``."""
    return [token for token in re.split(r"[^a-z0-9]+", name.lower()) if token]


def _matching_mmproj(model: Path, candidates: Sequence[Path]) -> list[Path]:
    """The projector(s) that plausibly belong to ``model`` among the
    ``mmproj*.gguf`` files beside it.

    A lone candidate is taken as is (unsloth ships Gemma's projector as
    the generic ``mmproj-F16.gguf``). Among several, a candidate must
    carry the model's family token — the stem's first alphanumeric run,
    ``qwen3vl`` or ``gemma`` — and the one sharing the most name tokens
    with the model wins (``...-8B-Instruct-...`` over ``...-4B-...``).
    An empty result means none is the model's; more than one, a tie
    (two quantizations of the same projector) the caller must not guess.
    """
    if len(candidates) <= 1:
        return list(candidates)
    model_tokens = _name_tokens(model.stem)
    if not model_tokens:
        return []
    family = model_tokens[0]
    scored: list[tuple[int, Path]] = []
    for candidate in candidates:
        if family not in candidate.stem.lower():
            continue
        tokens = set(_name_tokens(candidate.stem))
        scored.append((sum(1 for token in model_tokens if token in tokens), candidate))
    if not scored:
        return []
    best = max(score for score, _ in scored)
    return [path for score, path in scored if score == best]


# ── the engine ───────────────────────────────────────────────────────


class LlamaServerVisionLanguageEngine(VisionLanguageEngine):
    """Scriptwriter served by ``llama-server`` in a separate process.

    Spawn mode (default): the engine starts the server lazily on the
    first panel and stops it on ``release()``. Attach mode
    (``server_url``): the engine only sends requests.
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        mmproj_path: str | os.PathLike[str] | None = None,
        server_binary: str | os.PathLike[str] | None = None,
        server_url: str | None = None,
        host: str = DEFAULT_HOST,
        port: int | None = DEFAULT_PORT,
        n_gpu_layers: int = 99,
        ctx_size: int = DEFAULT_CTX_SIZE,
        extra_server_args: Sequence[str] = (),
        startup_timeout_s: float = 180.0,
        known_characters: Sequence[str] = DEFAULT_CAST,
        sfx_tags: Sequence[str] = (),
        sfx_intensity: int = 2,
        style_addendum: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.2,
        max_image_edge: int = DEFAULT_MAX_IMAGE_EDGE,
        use_json_schema: bool = True,
        log_path: str | os.PathLike[str] | None = None,
        client: Any = None,
        launcher: Launcher | None = None,
        health_check: HealthCheck | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        Args:
            model_path: GGUF weights (spawn mode). In attach mode it is
                only a label — a GGUF path or an Ollama tag — sent as the
                request's ``model`` field.
            mmproj_path: Multimodal projector GGUF. ``None`` looks for a
                ``mmproj*.gguf`` next to the model and warns if none.
            server_binary: ``llama-server`` executable; defaults to
                ``LLAMA_SERVER_BIN``, then ``llama-server`` on PATH.
            server_url: Attach to this already-running server instead of
                spawning one (``release()`` then frees nothing).
            host / port: Where the spawned server listens; ``port=None``
                picks a free port (the reviewer then cannot find it).
            n_gpu_layers: ``-ngl``; 99 = everything on the GPU.
            ctx_size: ``-c``; the KV-cache budget.
            extra_server_args: Appended verbatim to the command line
                (e.g. ``("--reasoning", "off")`` for Gemma 4).
            startup_timeout_s: How long to wait for ``/health``.
            known_characters / sfx_tags / sfx_intensity / style_addendum:
                Prompt inputs shared with every other scriptwriter.
            max_tokens: Output budget per panel.
            temperature: Sampling temperature (low: the output is JSON).
            max_image_edge: Long-edge downscale cap before upload.
            use_json_schema: Ask for grammar-constrained JSON; falls back
                to the prompt-only instruction if the server rejects it.
            log_path: Where the server's stdout/stderr go (default: the
                temp dir).
            client / launcher / health_check / sleep / clock: Injection
                points for tests; ``None`` builds the real ones lazily.
        """
        self.provider = "llamacpp"
        self._model_ref = str(model_path) if model_path else None
        self.model = model_label(model_path)
        self._mmproj_explicit = Path(mmproj_path) if mmproj_path else None
        self._mmproj_resolved: Path | None = None
        self._binary_explicit = Path(server_binary) if server_binary else None
        self._binary_resolved: Path | None = None
        self._server_url = _normalize_url(server_url) if server_url else None
        self._host = host
        self._port = port
        self._n_gpu_layers = n_gpu_layers
        self._ctx_size = ctx_size
        self._extra_server_args = tuple(str(arg) for arg in extra_server_args)
        self._startup_timeout_s = float(startup_timeout_s)
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_image_edge = max_image_edge
        self._use_json_schema = use_json_schema
        self._schema_supported = True  # flipped off once the server rejects it
        self._log_path = Path(log_path) if log_path else None
        self._client = client
        self._client_owned = client is None
        self._launcher = launcher
        self._health_check: HealthCheck = health_check or default_health_check
        self._sleep = sleep
        self._clock = clock
        self._proc: Any = None
        self._proc_url: str | None = None
        self._log_handle: Any = None
        self._finalizer: weakref.finalize | None = None
        self._system_prompt = (
            build_scriptwriter_prompt(known_characters, sfx_tags, sfx_intensity, style_addendum)
            + JSON_INSTRUCTION
        )
        self._schema = build_panel_schema(sfx_tags)

    # ── public surface ───────────────────────────────────────────────

    @property
    def attached(self) -> bool:
        """Whether the engine uses an external server it must not stop."""
        return self._server_url is not None

    @property
    def server_url(self) -> str | None:
        """Base URL of the server (``None`` until a free port is chosen)."""
        if self._server_url is not None:
            return self._server_url
        if self._proc_url is not None:
            return self._proc_url
        if self._port is not None:
            return f"http://{self._url_host()}:{self._port}"
        return None

    @property
    def model_path(self) -> Path | None:
        if self._model_ref is None:
            return None
        if self.attached and not self._model_ref.lower().endswith(".gguf"):
            return None  # an Ollama tag, not a file
        return Path(self._model_ref)

    @property
    def mmproj_path(self) -> Path | None:
        return self._mmproj_explicit or self._mmproj_resolved

    @property
    def server_binary(self) -> Path | None:
        return self._binary_resolved or self._binary_explicit

    @property
    def extra_server_args(self) -> tuple[str, ...]:
        return self._extra_server_args

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def fingerprint(self) -> str:
        """Checkpoint identity: model + prompt (+ JSON mechanism)."""
        digest = hashlib.sha1(self._system_prompt.encode("utf-8")).hexdigest()[:8]
        mode = "schema" if self._use_json_schema else "prompt"
        return f"vlm-llamacpp:{self.model}:prompt={digest}:json={mode}"

    def preflight(self) -> None:
        """Fail fast, before any pipeline work, with an actionable message.

        Both modes need the OpenAI SDK (an optional extra a fully local
        install may lack). Spawn mode also checks the binary and the
        GGUF files and reports every gap in one error; attach mode
        checks the server answers ``/health``.
        """
        if self.attached:
            self._require_openai_sdk()
            if not self._health_check(self._server_url):  # type: ignore[arg-type]
                raise RuntimeError(
                    f"Nenhum servidor respondeu em {self._server_url}/health. Inicie o "
                    f"llama-server (ou o Ollama) nesse endereço, ou remova {ENV_SERVER_URL} "
                    "para que o MangaWhisperer inicie o servidor sozinho."
                )
            return
        problems: list[str] = []
        model: Path | None = None
        try:
            self._require_openai_sdk()
        except RuntimeError as exc:
            problems.append(str(exc))
        try:
            model = self._resolve_model()
        except RuntimeError as exc:
            problems.append(str(exc))
        try:
            self._resolve_binary()
        except RuntimeError as exc:
            problems.append(str(exc))
        try:
            self._resolve_mmproj(model)
        except RuntimeError as exc:
            problems.append(str(exc))
        if problems:
            raise RuntimeError(
                "Não foi possível preparar o llama-server:\n- " + "\n- ".join(problems)
            )

    def contextualize(
        self, panel_image: Image, bubbles: list[SpeechBubble]
    ) -> list[ContextualizedBlock]:
        """Produce the ordered narration script for one panel.

        A server that cannot be started — or talked to, for want of the
        SDK — is fatal (nothing would work); a failed *request* degrades
        that panel to passthrough blocks so a volume-length run never
        aborts on one bad panel.
        """
        url = self._ensure_server()
        client = self._get_client(url)
        try:
            raw = self._request(client, panel_image, bubble_request_text(bubbles))
        except Exception as exc:
            logger.error(
                "llama-server falhou neste painel (%s); usando passthrough para %d balões",
                exc,
                len(bubbles),
            )
            return passthrough_blocks(bubbles)

        blocks = parse_script_blocks(raw)
        if blocks is None or (not blocks and bubbles):
            logger.warning(
                "llama-server/%s output yielded no usable blocks (%r...); falling back to "
                "passthrough for %d bubbles",
                self.model,
                raw[:80],
                len(bubbles),
            )
            return passthrough_blocks(bubbles)
        return blocks

    def release(self) -> None:
        """Stop the spawned server so its VRAM is free for the TTS stage.

        Idempotent; the next ``contextualize`` starts the server again.
        In attach mode nothing is stopped — the memory stays with the
        external server, which is logged loudly.
        """
        if self.attached:
            logger.warning(
                "Servidor externo em %s: release() não encerra nada e a VRAM continua com "
                "ele. Em GPUs de 8 GB, pare esse servidor antes da síntese de voz — ou "
                "remova %s para que o MangaWhisperer gerencie o llama-server.",
                self._server_url,
                ENV_SERVER_URL,
            )
            return
        if self._proc is None:
            return
        logger.info("Encerrando llama-server em %s", self._proc_url)
        self._cleanup_process()

    # ── server lifecycle ─────────────────────────────────────────────

    def _ensure_server(self) -> str:
        """Return the base URL of a live server, spawning one if needed."""
        if self.attached:
            return self._server_url  # type: ignore[return-value]
        if self._proc is not None:
            code = self._proc.poll()
            if code is None:
                return self._proc_url  # type: ignore[return-value]
            logger.warning("llama-server encerrou inesperadamente (código %s); reiniciando", code)
            self._cleanup_process()
        self.preflight()
        return self._spawn()

    def _spawn(self) -> str:
        port = self._port if self._port is not None else _free_port()
        url = f"http://{self._url_host()}:{port}"
        if self._health_check(url):
            raise RuntimeError(
                f"Já existe um servidor respondendo em {url}. Para usá-lo, defina "
                f"{ENV_SERVER_URL}={url}; para iniciar outro, encerre-o ou escolha outra "
                "porta (port=...)."
            )
        args = self._server_args(port)
        logger.info("Iniciando llama-server: %s", subprocess.list2cmdline(args))
        proc = (self._launcher or self._default_launcher)(args)
        self._proc, self._proc_url = proc, url
        self._finalizer = weakref.finalize(self, _terminate_process, proc)
        try:
            self._wait_until_ready(proc, url, args, port)
        except BaseException:
            self._cleanup_process()
            raise
        logger.info("llama-server pronto em %s (modelo %s)", url, self.model)
        return url

    def _wait_until_ready(self, proc: Any, url: str, args: list[str], port: int) -> None:
        start = self._clock()
        while True:
            code = proc.poll()
            if code is not None:
                raise RuntimeError(
                    f"llama-server encerrou durante a inicialização (código {code}). Causas "
                    f"comuns: porta {port} ocupada, modelo/mmproj incompatíveis com o binário, "
                    f"VRAM insuficiente. Veja o log em {self._resolve_log_path()} e o comando: "
                    f"{subprocess.list2cmdline(args)}"
                )
            if self._health_check(url):
                return
            if self._clock() - start >= self._startup_timeout_s:
                raise RuntimeError(
                    f"llama-server não respondeu em {url}/health após "
                    f"{self._startup_timeout_s:.0f} s. O modelo pode não caber na VRAM (em 8 GB "
                    "use Q4_K_M e um -c menor) ou o binário pode ser incompatível com a GPU. "
                    f"Veja o log em {self._resolve_log_path()}; comando: "
                    f"{subprocess.list2cmdline(args)}"
                )
            self._sleep(_HEALTH_POLL_INTERVAL_S)

    def _server_args(self, port: int) -> list[str]:
        args = [str(self.server_binary), "-m", str(self.model_path)]
        if self.mmproj_path is not None:
            args += ["--mmproj", str(self.mmproj_path)]
        args += [
            "--host", self._host,
            "--port", str(port),
            "-ngl", str(self._n_gpu_layers),
            "-c", str(self._ctx_size),
            "-np", "1",  # one slot: the whole context for the current panel
            "--jinja",  # chat template from the GGUF (default on new builds; harmless)
        ]
        args += list(self._extra_server_args)
        return args

    def _default_launcher(self, args: list[str]) -> Any:
        log_path = self._resolve_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = open(log_path, "ab")  # noqa: SIM115 — closed in _cleanup_process
        try:
            return subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            self._log_handle.close()
            self._log_handle = None
            raise RuntimeError(f"Não foi possível executar {args[0]}: {exc}") from exc

    def _cleanup_process(self) -> None:
        proc, self._proc = self._proc, None
        self._proc_url = None
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None
        if proc is not None:
            _terminate_process(proc)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        if self._client_owned:
            self._client = None  # the next server may sit on another port

    def _resolve_log_path(self) -> Path:
        if self._log_path is not None:
            return self._log_path
        return Path(tempfile.gettempdir()) / "mangawhisperer" / "llama-server.log"

    def _url_host(self) -> str:
        return DEFAULT_HOST if self._host in ("", "0.0.0.0", "::") else self._host

    # ── preflight pieces ─────────────────────────────────────────────

    def _require_openai_sdk(self) -> None:
        """An injected client needs no SDK; otherwise the package must be
        importable — found out here, not after the 5 GB server is up."""
        if self._client is not None:
            return
        try:
            missing = importlib.util.find_spec("openai") is None
        except ValueError:  # a module object without __spec__: importable anyway
            missing = False
        if missing:
            raise RuntimeError(_OPENAI_HINT)

    def _resolve_binary(self) -> Path:
        if self._binary_resolved is not None:
            return self._binary_resolved
        tried: list[str] = []
        for candidate in (self._binary_explicit, os.environ.get(ENV_SERVER_BIN)):
            if not candidate:
                continue
            path = Path(candidate)
            tried.append(str(path))
            if path.is_file():
                self._binary_resolved = path
                return path
            found = shutil.which(str(path))
            if found:
                self._binary_resolved = Path(found)
                return self._binary_resolved
        found = shutil.which(DEFAULT_BINARY)
        if found:
            self._binary_resolved = Path(found)
            return self._binary_resolved
        tried.append(f"'{DEFAULT_BINARY}' no PATH")
        raise RuntimeError(
            f"llama-server não encontrado (procurado em: {', '.join(tried)}). {_INSTALL_HINT}"
        )

    def _resolve_model(self) -> Path:
        if self._model_ref is None:
            raise RuntimeError(
                f"Nenhum modelo GGUF configurado: defina {ENV_MODEL_GGUF}=<caminho do .gguf> "
                "(ou passe --vlm-model <caminho>.gguf). Para usar um servidor já em execução, "
                f"defina {ENV_SERVER_URL}."
            )
        path = Path(self._model_ref)
        if not path.is_file():
            raise RuntimeError(
                f"Modelo GGUF não encontrado: {path}. Confira {ENV_MODEL_GGUF} (ou --vlm-model). "
                "Repositórios com GGUF + mmproj: Qwen/Qwen3-VL-8B-Instruct-GGUF, "
                "unsloth/gemma-4-E4B-it-GGUF."
            )
        return path

    def _resolve_mmproj(self, model: Path | None) -> Path | None:
        """Explicit mmproj must exist; otherwise look next to the model
        (``None`` model = nothing to search: the model error says it all).

        With several ``mmproj*.gguf`` beside the model — two model
        families downloaded into one folder — only the one whose name
        matches the model is taken; an ambiguous folder is an error
        rather than a guess, because a projector from another model
        makes the server die at startup or, worse, load and produce
        garbage the tolerant parser accepts.
        """
        if self._mmproj_explicit is not None:
            if not self._mmproj_explicit.is_file():
                raise RuntimeError(
                    f"Projetor multimodal (mmproj) não encontrado: {self._mmproj_explicit}. "
                    f"Confira {ENV_MMPROJ_GGUF} — é o arquivo mmproj-*.gguf do mesmo "
                    "repositório do modelo."
                )
            return self._mmproj_explicit
        if model is None:
            return None
        if self._mmproj_resolved is not None and self._mmproj_resolved.is_file():
            return self._mmproj_resolved
        siblings = sorted(p for p in model.parent.glob("mmproj*.gguf") if p.is_file())
        if siblings:
            matches = _matching_mmproj(model, siblings)
            if len(matches) == 1:
                self._mmproj_resolved = matches[0]
                logger.info("mmproj encontrado ao lado do modelo: %s", matches[0])
                return self._mmproj_resolved
            names = ", ".join(p.name for p in siblings)
            if matches:
                raise RuntimeError(
                    f"Mais de um mmproj ao lado de {model.name} combina com ele ({names}): "
                    f"defina {ENV_MMPROJ_GGUF}=<caminho> para escolher qual usar."
                )
            raise RuntimeError(
                f"Nenhum dos mmproj ao lado de {model.name} tem o nome do modelo ({names}), "
                f"e um projetor de outro modelo faz o servidor falhar ou gerar lixo: defina "
                f"{ENV_MMPROJ_GGUF}=<caminho> com o mmproj-*.gguf do mesmo repositório do modelo."
            )
        logger.warning(
            "Nenhum mmproj configurado nem encontrado ao lado de %s: o llama-server sobe SEM "
            "visão e cada painel degrada para passthrough. Baixe o mmproj-*.gguf do mesmo "
            "repositório e defina %s.",
            model,
            ENV_MMPROJ_GGUF,
        )
        return None

    # ── request ──────────────────────────────────────────────────────

    def _request(self, client: Any, panel_image: Image, request_text: str) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": png_data_url(panel_image, self._max_image_edge)},
                        },
                        {"type": "text", "text": request_text},
                    ],
                },
            ],
        }
        if not (self._use_json_schema and self._schema_supported):
            return self._complete(client, request)
        try:
            return self._complete(client, {**request, "response_format": self._response_format()})
        except Exception as exc:
            if not _looks_like_bad_request(exc):
                raise
            rejection = exc
        # A 400 alone does not blame the schema (context size, text-only
        # server, bad image...). If the plain request fails too, that error
        # propagates and the schema stays on for the next panel; only a
        # plain request that succeeds proves the constraint was refused.
        logger.debug("Requisição com response_format falhou (%s); repetindo sem ele", rejection)
        raw = self._complete(client, request)
        logger.warning(
            "O servidor rejeitou response_format/json_schema (%s); seguindo só com a "
            "instrução JSON do prompt para o resto da execução",
            rejection,
        )
        self._schema_supported = False
        return raw

    def _response_format(self) -> dict[str, Any]:
        # The OpenAI nested shape: llama-server reads json_schema.schema
        # (the README's flat example does not match its parser).
        return {"type": "json_schema", "json_schema": {"name": "panel_script", "schema": self._schema}}

    @staticmethod
    def _complete(client: Any, request: dict[str, Any]) -> str:
        response = client.chat.completions.create(**request)
        return response.choices[0].message.content or ""

    def _get_client(self, url: str) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI  # noqa: PLC0415 — deferred so tests need no SDK
            except ImportError as exc:
                raise RuntimeError(_OPENAI_HINT) from exc

            self._client = OpenAI(base_url=url + "/v1", api_key="sk-local")
        return self._client
