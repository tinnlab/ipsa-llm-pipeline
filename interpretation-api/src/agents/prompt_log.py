"""
Per-run logging of every LLM prompt and raw completion.

The pipeline used to keep no record of what was sent to or returned by the model: the
payload was built and posted and nothing wrote it down. `step{N}_output.json` holds only the
*parsed* result, so the byte-level prompt and the raw completion — the two things needed to
settle a reproducibility question or debug a bad generation — were gone. This module is the
side channel that keeps them.

Layout, one directory per report::

    <log root>/<studyId>/<analysisId>/<reportId>/
        run.json          # ids, job_id, provider/model config, start + end
        llm_calls.jsonl   # one JSON object per LLM call, append-only

One appended JSONL rather than a file per call: the file count stays fixed whatever the call
volume (a job makes tens of calls), an append-only file truncates at most its last line if the
process dies mid-run, cross-step ordering is inherent, and `jq` slices it.

Identity comes from a ContextVar rather than a constructor argument, because the step classes
build their LLM clients in `__init__` — long before a job exists — and ~30 call sites reach
those clients. The orchestrator sets the context per job and narrows it per step; the client
reads it at call time, so no call site changes.

Everything here is best-effort. Logging is a side channel: a failure to write a log line must
never fail a pipeline job, so every filesystem operation is guarded and warns at most once per
process.
"""

import json
import re
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import logging

from ..config import settings

logger = logging.getLogger(__name__)

CALLS_FILENAME = 'llm_calls.jsonl'
MANIFEST_FILENAME = 'run.json'

# Path component fallbacks. Depth is always three, so a partial set of ids still buckets
# usefully and tooling never has to special-case a shallow path.
NO_STUDY = '_no-study'
NO_ANALYSIS = '_no-analysis'
NO_REPORT = '_no-report'

# Ids arrive from an HTTP caller and become directory names, so treat them as hostile.
# Anything outside this set collapses to '_'; a value like '../../etc' therefore becomes one
# inert component rather than a traversal.
_UNSAFE = re.compile(r'[^A-Za-z0-9._-]')
_MAX_COMPONENT = 128

# Accept both spellings so an integration typo can't silently drop a whole run's logs.
_ID_KEYS = {
    'study_id': ('studyId', 'study_id'),
    'analysis_id': ('analysisId', 'analysis_id'),
    'report_id': ('reportId', 'report_id'),
}

_write_lock = threading.Lock()
_seq_lock = threading.Lock()
_seq_counters: Dict[str, int] = {}

# Warn once per process per failure kind rather than once per call: a broken log root would
# otherwise produce a warning for every LLM call in every job.
_warned: set = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message)


@dataclass(frozen=True)
class LogContext:
    """Who a given LLM call belongs to. Set by the orchestrator, read by the client."""
    study_id: Optional[str] = None
    analysis_id: Optional[str] = None
    report_id: Optional[str] = None
    job_id: Optional[str] = None
    step: Optional[str] = None


_ctx: ContextVar[LogContext] = ContextVar('llm_prompt_log_ctx', default=LogContext())


def current_context() -> LogContext:
    return _ctx.get()


@contextmanager
def log_context(**overrides: Optional[str]) -> Iterator[LogContext]:
    """Merge `overrides` onto the current context for the duration of the block.

    Merging rather than replacing lets the orchestrator set the ids once per job and then
    narrow only `step` per step. `None` values are ignored so a caller can pass an absent id
    without clobbering one already in scope.
    """
    merged = replace(_ctx.get(), **{k: v for k, v in overrides.items() if v is not None})
    token = _ctx.set(merged)
    try:
        yield merged
    finally:
        _ctx.reset(token)


def set_step(step: Optional[str]) -> None:
    """Narrow the current context to a step, without opening a block.

    The orchestrator runs its steps sequentially in one long method; setting the step in place
    keeps that readable instead of wrapping each 80-line step body in a `with`. Safe because
    the enclosing `log_context()` for the job restores the whole context when the run ends.
    """
    _ctx.set(replace(_ctx.get(), step=step))


def extract_ids(metadata: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """Pull the routing ids out of a pipeline input's free-form `metadata`.

    `metadata` is documented as free-form and is already plumbed end to end, so the ids ride
    along with no schema change. Only strings are accepted — a caller sending a dict or a list
    for `studyId` gets the fallback bucket rather than a stringified mess in a path.
    """
    ids: Dict[str, Optional[str]] = {'study_id': None, 'analysis_id': None, 'report_id': None}
    if not isinstance(metadata, dict):
        return ids
    for field, keys in _ID_KEYS.items():
        for key in keys:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                ids[field] = value
                break
    return ids


def strip_ids(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return `metadata` without the routing ids.

    This is why it exists: `orchestrator.py` copies every non-`organism` metadata key into the
    `context` dict that steps render straight into their prompts, and step 6 builds the report
    title from raw metadata keys. Leaving the ids in would inject opaque database ids into
    every prompt and into the rendered report — changing model input under the banner of
    observability, which is the opposite of the point. Strip once, here, instead of chasing
    eight separate `skip_fields` sets.
    """
    if not isinstance(metadata, dict):
        return {} if metadata is None else metadata
    drop = {k for keys in _ID_KEYS.values() for k in keys}
    return {k: v for k, v in metadata.items() if k not in drop}


def _component(value: Optional[str], fallback: str) -> str:
    """Reduce an untrusted id to one safe path component."""
    if not isinstance(value, str):
        return fallback
    cleaned = _UNSAFE.sub('_', value.strip())[:_MAX_COMPONENT]
    # Strip dots so '..' can never survive as a component in its own right.
    cleaned = cleaned.strip('. ')
    return cleaned or fallback


def log_root() -> Path:
    """The configured log root, as an absolute path."""
    return Path(settings.LLM_PROMPT_LOG_DIR).expanduser().resolve()


def enabled() -> bool:
    return bool(settings.LLM_PROMPT_LOG_ENABLED)


def resolve_dir(ctx: Optional[LogContext] = None) -> Optional[Path]:
    """Directory for this run's logs, or None if logging is off or the path is unusable.

    Falls back per level, so a caller that sends only some of the ids still gets a useful
    bucket, and a run with no ids at all lands under its job_id.
    """
    if not enabled():
        return None

    ctx = ctx if ctx is not None else _ctx.get()
    root = log_root()

    study = _component(ctx.study_id, NO_STUDY)
    analysis = _component(ctx.analysis_id, NO_ANALYSIS)
    report = _component(ctx.report_id, _component(ctx.job_id, NO_REPORT))

    candidate = (root / study / analysis / report).resolve()

    # Belt and braces behind the sanitiser: whatever the components turned out to be, the
    # result has to sit under the root. If it somehow doesn't, bucket the run by job_id
    # rather than writing outside the log root.
    try:
        candidate.relative_to(root)
    except ValueError:
        _warn_once(
            'escape',
            'LLM prompt log: resolved path escaped the log root; using the job_id bucket instead'
        )
        candidate = (root / NO_STUDY / NO_ANALYSIS / _component(ctx.job_id, NO_REPORT)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
    return candidate


def _next_seq(directory: Path) -> int:
    """Next call number for a run.

    Seeded from the lines already in the file the first time this process sees a directory, so
    a service restart part-way through a long run continues the numbering instead of writing a
    second call #1 into the same file.
    """
    with _seq_lock:
        key = str(directory)
        if key not in _seq_counters:
            existing = 0
            try:
                target = directory / CALLS_FILENAME
                if target.exists():
                    with open(target, 'r', encoding='utf-8') as handle:
                        existing = sum(1 for line in handle if line.strip())
            except Exception:
                existing = 0
            _seq_counters[key] = existing
        nxt = _seq_counters[key] + 1
        _seq_counters[key] = nxt
        return nxt


def start_record(
    *,
    provider: str,
    role: str,
    model: str,
    endpoint: str,
    method: str,
    stream: bool,
    payload: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Begin a call record. Returns None when logging is off — callers treat that as a no-op."""
    if not enabled():
        return None
    try:
        ctx = _ctx.get()
        return {
            '_started': time.time(),
            'call_id': str(uuid.uuid4()),
            'ts_start': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'job_id': ctx.job_id,
            'study_id': ctx.study_id,
            'analysis_id': ctx.analysis_id,
            'report_id': ctx.report_id,
            'step': ctx.step,
            'role': role,
            'provider': provider,
            'method': method,
            'stream': stream,
            'model': model,
            'endpoint': endpoint,
            'request': payload,
            'http_status': None,
            'response_body': None,
            'response_text': None,
            'usage': None,
            'error': None,
        }
    except Exception as exc:  # pragma: no cover - defensive
        _warn_once('start', f'LLM prompt log: could not start a record ({exc})')
        return None


def write(record: Optional[Dict[str, Any]]) -> None:
    """Append one finished record. Never raises."""
    if not record:
        return
    try:
        # Resolve from the record's OWN ids, not from whatever the ambient context happens to
        # be now. They are the same inside a single call today, but a record must always land
        # in the run it belongs to even if the context moves on before the write.
        directory = resolve_dir(LogContext(
            study_id=record.get('study_id'),
            analysis_id=record.get('analysis_id'),
            report_id=record.get('report_id'),
            job_id=record.get('job_id'),
            step=record.get('step'),
        ))
        if directory is None:
            return
        directory.mkdir(parents=True, exist_ok=True)

        started = record.pop('_started', None)
        if started is not None and 'duration_ms' not in record:
            record['duration_ms'] = int((time.time() - started) * 1000)
        record['seq'] = _next_seq(directory)

        line = json.dumps(record, ensure_ascii=False, default=str)
        target = directory / CALLS_FILENAME
        with _write_lock:
            with open(target, 'a', encoding='utf-8') as handle:
                handle.write(line + '\n')
    except Exception as exc:
        _warn_once('write', f'LLM prompt log: could not write a call record ({exc})')


def write_manifest(extra: Optional[Dict[str, Any]] = None) -> None:
    """Write/refresh `run.json` for the current context. Never raises."""
    try:
        directory = resolve_dir()
        if directory is None:
            return
        directory.mkdir(parents=True, exist_ok=True)

        ctx = _ctx.get()
        target = directory / MANIFEST_FILENAME
        manifest: Dict[str, Any] = {}
        if target.exists():
            try:
                manifest = json.loads(target.read_text(encoding='utf-8'))
            except Exception:
                manifest = {}

        manifest.update({
            'job_id': ctx.job_id,
            'study_id': ctx.study_id,
            'analysis_id': ctx.analysis_id,
            'report_id': ctx.report_id,
            'updated_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        })
        manifest.setdefault(
            'started_at', datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'))
        if extra:
            manifest.update(extra)

        target.write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')
    except Exception as exc:
        _warn_once('manifest', f'LLM prompt log: could not write the run manifest ({exc})')


def prune(root: Optional[Path] = None, days: Optional[int] = None) -> int:
    """Delete report directories older than `days`, then tidy the levels above them.

    An unbounded log directory on a shared host is its own problem. Age-based, run at service
    start, confined to the log root by the same containment check used for writes. `days <= 0`
    means keep forever. Returns the number of report directories removed.
    """
    removed = 0
    try:
        base = (root or log_root())
        limit = settings.LLM_PROMPT_LOG_RETENTION_DAYS if days is None else days
        if limit is None or limit <= 0 or not base.is_dir():
            return 0

        base = base.resolve()
        cutoff = time.time() - (limit * 86400)

        for study_dir in [p for p in base.iterdir() if p.is_dir()]:
            for analysis_dir in [p for p in study_dir.iterdir() if p.is_dir()]:
                for report_dir in [p for p in analysis_dir.iterdir() if p.is_dir()]:
                    try:
                        resolved = report_dir.resolve()
                        resolved.relative_to(base)
                    except ValueError:
                        continue  # never follow a link out of the root
                    if resolved.stat().st_mtime >= cutoff:
                        continue
                    shutil.rmtree(resolved)
                    removed += 1
                _rmdir_if_empty(analysis_dir)
            _rmdir_if_empty(study_dir)

        if removed:
            logger.info(
                f'LLM prompt log: pruned {removed} run(s) older than {limit} day(s) from {base}')
    except Exception as exc:
        _warn_once('prune', f'LLM prompt log: retention pass failed ({exc})')
    return removed


def _rmdir_if_empty(directory: Path) -> None:
    try:
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    except OSError:
        pass


def describe_root() -> str:
    """Human-readable statement of where logs are going, for the startup banner.

    Worth logging because the worst failure mode of this feature is silent: if the configured
    directory is not the mounted volume, logging keeps working, writes into the container's
    writable layer, and every redeploy wipes it with no error anywhere.
    """
    if not enabled():
        return 'LLM prompt logging is DISABLED (LLM_PROMPT_LOG_ENABLED=false)'
    retention = settings.LLM_PROMPT_LOG_RETENTION_DAYS
    horizon = f'{retention} day(s)' if retention and retention > 0 else 'forever (no pruning)'
    return f'LLM prompt logging -> {log_root()} (retention: {horizon})'


def log_startup_banner() -> None:
    """Announce the resolved log root through `logging`, not `print`.

    Deliberately not a print: stdout is block-buffered when the service runs under uvicorn in a
    container (nothing sets PYTHONUNBUFFERED), so a printed banner can sit unflushed for
    minutes — useless for the `docker compose logs` check that is meant to confirm the logs are
    going somewhere persistent.
    """
    try:
        logger.info(describe_root())
    except Exception:  # pragma: no cover - defensive
        pass
