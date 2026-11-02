"""HTTP-independent validation, startup and serialized inference."""
import json
import threading
import time
import uuid
from pathlib import Path

from backends import StubBackend, TransformersBackend
import journal
import release


class RequestError(ValueError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


class ServiceState:
    def __init__(self, root, warmup_seconds=0, backend_kind='stub'):
        self.root = Path(root)
        self.manifest = journal.read(self.root / 'accepted.json')
        self.backend = None
        self.error = None
        self.warmup_finished = threading.Event()
        self.compute = threading.Lock()
        self.inflight = threading.BoundedSemaphore(8)
        self.backend_kind = backend_kind
        self.thread = threading.Thread(target=self._load, args=(warmup_seconds,), daemon=True)
        self.thread.start()

    @property
    def ready(self):
        return self.warmup_finished.is_set() and self.error is None

    def _load(self, seconds):
        try:
            manifest = self.manifest
            release.verify_artifact(manifest, journal.artifact_root(self.root, manifest))
            synthetic = manifest['runtime'].get('base_revision') == 'fixture-only'
            if synthetic != (self.backend_kind == 'stub'):
                raise ValueError('Stub / real artifact mismatch')
            if not synthetic:
                from lineage import verify_accepted
                verify_accepted(self.root, manifest)
            backend = StubBackend(manifest) if synthetic else TransformersBackend(manifest)
            time.sleep(seconds)
            backend.generate('In pandas, select the first row. Choices: A. df.iloc[0] B. df.iloc[1] C. df.iloc[2]',
                             manifest['runtime']['generation']['max_new_tokens'])
            self.backend = backend
            self.warmup_finished.set()
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            import logging
            logging.exception('Model loading/warmup failed')

    def debug(self):
        return dict(loaded_id=self.backend.loaded_id if self.backend else None,
                    calls=self.backend.calls if self.backend else 0,
                    warmup_finished=self.warmup_finished.is_set(), ready=self.ready)

    def validate(self, request):
        if not isinstance(request, dict) or set(request) - {'prompt', 'max_new_tokens'}:
            raise RequestError(422, 'Expected prompt and optional max_new_tokens')
        prompt = request.get('prompt')
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000:
            raise RequestError(422, 'prompt must contain 1..8000 characters')
        evaluated = self.manifest['runtime']['generation']['max_new_tokens']
        limit = request.get('max_new_tokens', evaluated)
        if type(limit) is not int or not 1 <= limit <= evaluated:
            raise RequestError(422, f'max_new_tokens must be an integer in 1..{evaluated}')
        if self.backend_kind != 'stub' and limit != evaluated:
            raise RequestError(422, f'Evaluated API requires max_new_tokens={evaluated}')
        return prompt, limit

    def generate(self, request):
        started = time.perf_counter()
        prompt, limit = self.validate(request)
        if not self.ready:
            raise RequestError(503, 'Model not ready')
        if not self.inflight.acquire(blocking=False):
            raise RequestError(429, 'Inference queue full')
        try:
            with self.compute:
                computing = time.perf_counter()
                try:
                    result = self.backend.generate(prompt, limit)
                except ValueError as exc:
                    raise RequestError(422, str(exc)) from exc
                finished = time.perf_counter()
        finally:
            self.inflight.release()
        result.update(model_id=self.backend.loaded_id, runtime_sha256=self.backend.runtime_sha256,
                      request_id=str(uuid.uuid4()), generation_seconds=finished-computing,
                      queue_seconds=computing-started, total_seconds=finished-started)
        return result
