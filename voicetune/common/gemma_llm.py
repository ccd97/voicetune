"""LLM helpers shared across pipeline stages."""

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

_llamacpp_cache: dict[tuple[str, int, str | None], Any] = {}


def load_llm(*, n_ctx: int, mmproj_path: str | None = None) -> Any:
    """Load and cache a llama.cpp model, keyed on (LLAMACPP_MODEL_PATH, n_ctx, mmproj_path)."""
    model_path = os.environ["LLAMACPP_MODEL_PATH"]
    key = (model_path, n_ctx, mmproj_path)
    if key in _llamacpp_cache:
        return _llamacpp_cache[key]

    from llama_cpp import Llama

    log.info(f"Loading llama.cpp model: {model_path}")
    handler = None
    if mmproj_path is not None:
        # No Gemma4ChatHandler in llama-cpp-python yet; Gemma3's works for the mmproj path.
        from llama_cpp.llama_chat_format import Gemma3ChatHandler
        log.info(f"mmproj: {mmproj_path}")
        handler = Gemma3ChatHandler(clip_model_path=mmproj_path, verbose=False)

    llm = Llama(
        model_path=model_path,
        chat_handler=handler,
        n_ctx=n_ctx,
        n_gpu_layers=-1,
        n_threads=os.cpu_count() or 4,
        verbose=False,
    )
    _llamacpp_cache[key] = llm
    return llm


def generate_text(llm, prompt: str, *, max_tokens: int, temperature: float = 0.0) -> str:
    """Single-turn text completion; returns the assistant's content string."""
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response["choices"][0]["message"]["content"]
