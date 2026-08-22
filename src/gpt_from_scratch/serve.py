import argparse
import uuid
from pathlib import Path
from typing import Any

import torch

from gpt_from_scratch.sample import check_stop, load_model, pick_device


def create_app(checkpoint_path: Path, *, device: torch.device | None = None) -> Any:
    """Build a FastAPI app exposing POST /v1/completions over a checkpoint.

    Requires the serve extra (fastapi); uvicorn is only needed by :func:`main`.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError(
            "serving requires fastapi; install it with: uv pip install -e '.[serve]'"
        ) from exc

    app = FastAPI(title="gpt-from-scratch", version="0.1.0")
    resolved_device = device if device is not None else pick_device()
    model, tokenizer = load_model(Path(checkpoint_path), resolved_device)
    app.state.model = model
    app.state.tokenizer = tokenizer

    class CompletionBody(BaseModel):
        prompt: str
        max_tokens: int = Field(default=64, ge=1)
        temperature: float = Field(default=1.0, gt=0)
        top_k: int | None = None
        top_p: float | None = None
        stop: list[str] | None = None

    @app.post("/v1/completions")
    def completions(body: CompletionBody) -> dict[str, Any]:
        prompt_ids = tokenizer.encode(body.prompt)
        if not prompt_ids:
            raise HTTPException(status_code=400, detail="prompt encodes to zero tokens")
        context = torch.tensor([prompt_ids], dtype=torch.long, device=resolved_device)
        full = model.generate(
            context,
            max_new_tokens=body.max_tokens,
            temperature=body.temperature,
            top_k=body.top_k,
            top_p=body.top_p,
            use_cache=True,
        )
        n_generated = int(full.shape[1] - len(prompt_ids))
        text = tokenizer.decode(full[0].tolist()[len(prompt_ids) :])
        finish_reason = "length"
        if body.stop:
            text, hit = check_stop(text, body.stop)
            if hit:
                finish_reason = "stop"
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "choices": [{"text": text, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": n_generated},
        }

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a checkpoint behind an OpenAI-style completions API"
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/checkpoint.pt"))
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    app = create_app(args.checkpoint)
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "serving requires uvicorn; install it with: uv pip install -e '.[serve]'"
        ) from exc
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
