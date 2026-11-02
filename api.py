"""FastAPI entry point with the same command/HTTP contract as service.py."""
import argparse
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api_core import RequestError, ServiceState


def create_app(root=Path('.runtime'), backend='transformers', warmup_seconds=0):
    @asynccontextmanager
    async def lifespan(app):
        app.state.model = ServiceState(root, warmup_seconds, backend)
        yield
        # Do not tear down a model while a worker is computing. uvicorn drains requests.

    app = FastAPI(title='Pandas model release API', lifespan=lifespan)

    @app.get('/health')
    def health():
        return {'alive': True, 'backend': backend}

    @app.get('/ready')
    def ready():
        state = app.state.model
        return JSONResponse(status_code=200 if state.ready else 503, content={
            'ready': state.ready, 'model_id': state.backend.loaded_id if state.ready else None,
            'runtime_sha256': state.backend.runtime_sha256 if state.ready else None,
            'load_failed': state.error is not None})

    @app.post('/generate')
    async def generate(request: Request):
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 65536:
                return JSONResponse(status_code=413, content={'error': 'Body too large'})
        try:
            import json
            payload = json.loads(body)
            result = await asyncio.to_thread(app.state.model.generate, payload)
            return result
        except RequestError as exc:
            return JSONResponse(status_code=exc.status, content={'error': str(exc)})
        except (ValueError, TypeError) as exc:
            return JSONResponse(status_code=422, content={'error': str(exc)})

    if backend == 'stub':
        @app.get('/debug')
        def debug():
            return app.state.model.debug()
    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path('.runtime'))
    parser.add_argument('--port', type=int, default=8091)
    parser.add_argument('--warmup-seconds', type=float, default=0)
    parser.add_argument('--backend', choices=['stub', 'transformers'], default='transformers')
    args = parser.parse_args()
    uvicorn.run(create_app(args.workspace, args.backend, args.warmup_seconds),
                host='127.0.0.1', port=args.port, workers=1)


if __name__ == '__main__':
    main()
