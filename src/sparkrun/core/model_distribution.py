"""Model transfer providers; manifests, validation and launch policy stay in core."""

from __future__ import annotations

from sparkrun.feedback import FeedbackReporter, get_feedback

from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import wraps
from inspect import signature
import logging
import math
from pathlib import PurePosixPath
from typing import Any, Callable, Protocol

from sparkrun.core.registration import enlist_registry_state, register_unique
from sparkrun.models.artifacts import ModelArtifactError, ModelArtifactManifest, ValidatedModelBinding
from sparkrun.transports.session import HostSession

logger = logging.getLogger(__name__)
MODEL_DISTRIBUTION_API_VERSION = 1
MODEL_PULL_API_VERSION = 1
_PROVIDERS: dict[str, ModelDistributionProvider] = {}
enlist_registry_state(globals(), "_PROVIDERS")
_CONFIG: ContextVar[Any] = ContextVar("model_distribution_config", default=None)
# Seconds without transferred bytes advancing before a copy is called stalled.
DEFAULT_MODEL_TRANSFER_IDLE_TIMEOUT_S = 600
DEFAULT_MODEL_TRANSFER_STALL_RETRIES = 1
# How long a preparation waits for another one's repository lease. Leases are
# held across acquisition, so a healthy wait can span someone else's whole
# multi-hundred-GB download (450 GiB at 20 MiB/s is 6.4 hours), and several
# launches may queue on one source. A day outlasts that while still failing a
# boot-started or scripted launch queued behind a holder that is truly stuck.
DEFAULT_MODEL_CACHE_LEASE_TIMEOUT_S = 24 * 3600
# What a provider receives when no total ceiling is configured. Providers
# predating the progress watchdog do arithmetic on request.timeout, so they
# get a large finite value rather than None.
PROVIDER_FALLBACK_TIMEOUT_S = 24 * 60 * 60
_BINDINGS: ContextVar[dict[tuple[str, str], ValidatedModelBinding] | None] = ContextVar("validated_models", default=None)


@dataclass(frozen=True)
class ModelTarget:
    host: str
    cache_root: str
    transfer_host: str
    required_files: tuple[str, ...] = ()

    def __post_init__(self):
        if any(
            not name or name.startswith("-") or any(c.isspace() for c in name) or "\x00" in name for name in (self.host, self.transfer_host)
        ):
            raise ModelArtifactError("model target must identify a valid management and transfer host")
        if not PurePosixPath(self.cache_root).is_absolute() or "\x00" in self.cache_root:
            raise ModelArtifactError("model provider cache roots must be prepared absolute paths")


@dataclass(frozen=True)
class ModelCopyRequest:
    feedback: FeedbackReporter | None = field(
        default_factory=lambda: get_feedback("model-distribution"), repr=False, compare=False, kw_only=True
    )
    manifest: ModelArtifactManifest
    source_host: str | None
    source_cache_root: str
    targets: tuple[ModelTarget, ...]
    operation_id: str
    offline: bool = False
    dry_run: bool = False
    # Total ceiling; None (the default) means none. A transfer is bounded by
    # its progress instead: idle_timeout seconds without a byte advancing
    # cancels it, however long the healthy part took (#324).
    timeout: float | None = None
    idle_timeout: float | None = field(default=DEFAULT_MODEL_TRANSFER_IDLE_TIMEOUT_S, kw_only=True)
    config: Any = field(default=None, repr=False, compare=False)
    session: HostSession | None = field(default=None, repr=False, compare=False)
    cancelled: Callable[[], bool] = field(default=lambda: False, repr=False, compare=False)
    progress: Callable[[str, int], None] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.targets, tuple) or len({t.host for t in self.targets}) != len(self.targets):
            raise ModelArtifactError("model provider targets must be immutable and unique")
        if not PurePosixPath(self.source_cache_root).is_absolute() or "\x00" in self.source_cache_root:
            raise ModelArtifactError("model provider source must be a prepared absolute path")
        paths = {f.path for f in self.manifest.files}
        for target in self.targets:
            if (
                not isinstance(target.required_files, tuple)
                or len(set(target.required_files)) != len(target.required_files)
                or set(target.required_files) - paths
            ):
                raise ModelArtifactError("model transfer files must be unique members of the manifest")
        if self.timeout is not None and (not math.isfinite(self.timeout) or self.timeout < 0):
            raise ModelArtifactError("model transfer timeout must be finite and nonnegative")
        if self.idle_timeout is not None and (not math.isfinite(self.idle_timeout) or self.idle_timeout <= 0):
            raise ModelArtifactError("model transfer idle timeout must be finite and positive")


@dataclass(frozen=True)
class ModelPullRequest(ModelCopyRequest):
    token: Callable[[], str | None] = field(default=lambda: None, repr=False, compare=False)


@dataclass(frozen=True)
class ModelTransferResult:
    manifest_id: str
    outcomes: dict[str, str]
    errors: dict[str, str] = field(default_factory=dict)


class ModelDistributionUnsupported(ModelArtifactError):
    """The request is unsupported, before any acquisition or cache mutation."""


class ModelDistributionFailed(ModelArtifactError):
    """A provider failed or violated its contract; automatic fallback is unsafe."""

    def __init__(self, message: str, *, failures: dict[str, str] | None = None):
        super().__init__(message)
        self.failures = failures or {}


class ModelDistributionProvider(Protocol):
    def copy(self, request: ModelCopyRequest) -> ModelTransferResult: ...


def register_model_distribution_provider(name: str, provider: ModelDistributionProvider) -> None:
    if not name or name in {"auto", "builtin"} or not callable(getattr(provider, "copy", None)):
        raise ValueError("model distribution provider needs a unique name and copy method")
    if getattr(provider, "api_version", MODEL_DISTRIBUTION_API_VERSION) != MODEL_DISTRIBUTION_API_VERSION:
        raise ValueError("unsupported model distribution provider API version")
    register_unique(_PROVIDERS, name, provider, description="model distribution provider")


def model_distribution_operation(function: Callable) -> Callable:
    call_signature = signature(function)

    @wraps(function)
    def scoped(*args, **kwargs):
        arguments = call_signature.bind(*args, **kwargs).arguments
        config = arguments.get("config", _CONFIG.get())
        if config is None:
            config = getattr(arguments.get("sctx"), "config", None)
        token = _CONFIG.set(config)
        bindings_token = _BINDINGS.set({}) if _BINDINGS.get() is None else None
        try:
            from sparkrun.models.usage import usage_operation

            with usage_operation():
                return function(*args, **kwargs)
        finally:
            _CONFIG.reset(token)
            if bindings_token is not None:
                _BINDINGS.reset(bindings_token)

    return scoped


def operation_config():
    return _CONFIG.get()


def _setting(config: Any, key: str, default: Any) -> Any:
    get = getattr(config, "get", None)
    return get(key, default) if callable(get) else default


@dataclass(frozen=True)
class TransferTimeouts:
    """How long a model transfer may run, from ``model_transfer_*`` config."""

    idle_timeout: float = DEFAULT_MODEL_TRANSFER_IDLE_TIMEOUT_S
    timeout: float | None = None
    stall_retries: int = DEFAULT_MODEL_TRANSFER_STALL_RETRIES


def transfer_timeouts(config=None) -> TransferTimeouts:
    """Resolve the progress watchdog settings.

    ``model_transfer_idle_timeout_s`` (positive, default 600) bounds a stall;
    ``model_transfer_timeout_s`` (0, the default, means none) is an opt-in
    total ceiling; ``model_transfer_stall_retries`` (default 1) resumes a
    stalled rsync that many times before failing.
    """
    config = _CONFIG.get() if config is None else config

    def number(key, default, *, allow_zero):
        value = _setting(config, key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ModelArtifactError(key + " must be a nonnegative number of seconds")
        if value == 0 and not allow_zero:
            raise ModelArtifactError(key + " must be positive")
        return value

    idle = number("model_transfer_idle_timeout_s", DEFAULT_MODEL_TRANSFER_IDLE_TIMEOUT_S, allow_zero=False)
    total = number("model_transfer_timeout_s", 0, allow_zero=True)
    retries = _setting(config, "model_transfer_stall_retries", DEFAULT_MODEL_TRANSFER_STALL_RETRIES)
    if type(retries) is not int or retries < 0:
        raise ModelArtifactError("model_transfer_stall_retries must be a nonnegative integer")
    return TransferTimeouts(idle, total or None, retries)


def lease_timeout(config=None) -> float | None:
    """Seconds to wait for a busy model cache lease, or ``None`` for no limit.

    ``model_cache_lease_timeout_s`` (cluster ``distribution.model.lease_timeout_s``),
    default :data:`DEFAULT_MODEL_CACHE_LEASE_TIMEOUT_S`; ``0`` waits until cancelled.
    """
    config = _CONFIG.get() if config is None else config
    value = _setting(config, "model_cache_lease_timeout_s", DEFAULT_MODEL_CACHE_LEASE_TIMEOUT_S)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ModelArtifactError("model_cache_lease_timeout_s must be a nonnegative number of seconds")
    return value or None


def validation_policy(config=None) -> str:
    config = _CONFIG.get() if config is None else config
    # Every policy prepares manifests. Legacy permits missing-inventory
    # compatibility; providers always require the strict inventory contract.
    policy = _setting(config, "model_cache_validation", "legacy")
    provider = _setting(config, "model_distribution_provider", "auto")
    if not isinstance(provider, str):
        raise ModelArtifactError("model_distribution_provider must be auto, builtin or a provider name")
    if not isinstance(policy, str) or policy not in {"legacy", "metadata", "checksum"}:
        raise ModelArtifactError("model_cache_validation must be legacy, metadata or checksum")
    if policy == "legacy" and (provider not in {"auto", "builtin"} or (provider == "auto" and _PROVIDERS)):
        return "metadata"
    return policy


def record_binding(binding: ValidatedModelBinding) -> None:
    if not binding.report.complete or binding.report.manifest_id != binding.manifest.identity:
        raise ModelArtifactError("only validated model bindings can reach the runtime")
    bindings = _BINDINGS.get()
    if bindings is not None:
        key = binding.model, binding.host
        old = bindings.get(key)
        if old is not None and old.manifest.identity != binding.manifest.identity:
            binding = _merge_filtered_bindings(old, binding)
        bindings[key] = binding


def _merge_filtered_bindings(old: ValidatedModelBinding, new: ValidatedModelBinding) -> ValidatedModelBinding:
    """Combine validated file scopes in one snapshot, never different revisions.

    The union is operation-local: persisted manifests retain their individual
    selection identities. A union of partial scopes remains marked as filtered.
    """
    left, right = old.manifest, new.manifest
    if not (left.has_file_patterns or right.has_file_patterns) or (left.repo, left.revision, left.endpoint, old.snapshot_path) != (
        right.repo,
        right.revision,
        right.endpoint,
        new.snapshot_path,
    ):
        raise ModelArtifactError("one host cannot bind the same model name to different artifacts")
    files = {entry.path: entry for entry in left.files}
    for entry in right.files:
        if entry.path in files and files[entry.path] != entry:
            raise ModelArtifactError("filtered model inventories disagree about " + entry.path)
        files[entry.path] = entry
    for name in ("weight_path", "projector_path"):
        a, b = getattr(left, name), getattr(right, name)
        if a is not None and b is not None and a != b:
            raise ModelArtifactError("filtered model inventories disagree about " + name)
    if not left.has_file_patterns and len(files) == len(left.files):
        return old
    if not right.has_file_patterns and len(files) == len(right.files):
        return new
    manifest = replace(
        left,
        selection="file-patterns-v1:union",
        files=tuple(files[path] for path in sorted(files)),
        weight_path=left.weight_path or right.weight_path,
        projector_path=left.projector_path or right.projector_path,
        weight_files=tuple(sorted(set(left.weight_files) | set(right.weight_files))),
    )
    report = replace(
        new.report,
        manifest_id=manifest.identity,
        checked_files=len(files),
        level="checksum" if old.report.level == new.report.level == "checksum" else "metadata",
    )
    return replace(new, manifest=manifest, report=report)


def prepared_model(model: str, host: str) -> ValidatedModelBinding | None:
    return (_BINDINGS.get() or {}).get((model, host))


def prepared_models() -> tuple[ValidatedModelBinding, ...]:
    return tuple((_BINDINGS.get() or {}).values())


def try_model_transfer(request: ModelCopyRequest, *, pull: bool = False) -> ModelTransferResult | None:
    """Return None for planned builtin work, never after partial provider work.

    A provider borrows the prepared session. Operation state is explicit in the
    request, including in callback threads; no ContextVar lookup is required.
    """
    selected = _setting(request.config, "model_distribution_provider", "auto")
    if not isinstance(selected, str):
        raise ModelArtifactError("model_distribution_provider must be auto, builtin or a provider name")
    if selected == "builtin" or (selected == "auto" and not _PROVIDERS):
        return None
    if selected == "auto":
        if len(_PROVIDERS) != 1:
            raise ModelArtifactError("multiple model providers enabled; set model_distribution_provider explicitly")
        selected = next(iter(_PROVIDERS))
    if selected not in _PROVIDERS:
        raise ModelArtifactError("model distribution provider is not enabled: " + str(selected))
    provider = _PROVIDERS[selected]
    method = getattr(provider, "pull" if pull else "copy", None)
    if not callable(method) and pull:
        return None  # Copy-only providers compose with builtin acquisition.
    if not callable(method):
        raise ModelDistributionFailed("model provider has no callable copy method")
    if request.offline and pull and getattr(provider, "supports_offline_pull", False) is not True:
        raise ModelDistributionUnsupported("selected model provider cannot acquire artifacts offline")
    if request.dry_run:
        return None  # Planning must not manufacture cache-readiness evidence.
    if request.cancelled():
        raise ModelDistributionFailed("model transfer cancelled")
    try:
        result = method(request if request.timeout is not None else replace(request, timeout=PROVIDER_FALLBACK_TIMEOUT_S))
    except ModelDistributionUnsupported as exc:
        fallback = _setting(request.config, "model_distribution_fallback", getattr(provider, "fallback_on_unsupported", False))
        if _setting(request.config, "model_distribution_provider", "auto") == "auto" and fallback is True:
            logger.warning("Model provider unavailable (%s); using builtin transfer", exc)
            return None
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        failures = {request.targets[0].host: str(exc)} if len(request.targets) == 1 else {}
        raise ModelDistributionFailed("model provider failed; no fallback after transfer starts", failures=failures) from exc
    if request.cancelled():
        raise ModelDistributionFailed("model transfer cancelled")
    hosts = {target.host for target in request.targets}
    if not isinstance(result, ModelTransferResult) or result.manifest_id != request.manifest.identity:
        raise ModelDistributionFailed("model provider returned a different artifact identity")
    if (
        not isinstance(result.outcomes, dict)
        or not isinstance(result.errors, dict)
        or any(not isinstance(state, str) for state in result.outcomes.values())
    ):
        raise ModelDistributionFailed("model provider returned malformed outcomes")
    if len(hosts) != len(request.targets) or set(result.outcomes) != hosts or set(result.errors) - hosts:
        raise ModelDistributionFailed("model provider must report exactly the requested targets")
    if any(state not in {"complete", "already_present", "failed"} for state in result.outcomes.values()):
        raise ModelDistributionFailed("model provider returned an invalid target state")
    if any(result.outcomes[h] != "failed" for h in result.errors):
        raise ModelDistributionFailed("model provider returned conflicting success/error evidence")
    if any(state == "failed" for state in result.outcomes.values()):
        failures = {h: result.errors.get(h, "transfer failed") for h, state in result.outcomes.items() if state == "failed"}
        raise ModelDistributionFailed(
            "model provider failed: " + "; ".join(h + ": " + reason for h, reason in failures.items()), failures=failures
        )
    return result
