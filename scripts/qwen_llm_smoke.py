from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.qwen3asr_dml.runtime import (
    Qwen3ASRDmlModel,
    resolve_encoder_filenames,
    resolve_llm_filename,
    resolve_model_dir,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Load and close the Qwen3 ASR DML runtime.")
    parser.add_argument("model_path", help="Path to the Qwen3-ASR DML model directory.")
    parser.add_argument("--device", default="dml", help="Runtime device, e.g. dml or cpu.")
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_path)
    llm_fn = resolve_llm_filename(model_dir)
    encoder_frontend_fn, encoder_backend_fn = resolve_encoder_filenames(model_dir)

    print(f"model_dir={model_dir}")
    print(f"llm_fn={llm_fn}")
    print(f"encoder_frontend_fn={encoder_frontend_fn}")
    print(f"encoder_backend_fn={encoder_backend_fn}")

    model = Qwen3ASRDmlModel.from_model_path(model_dir, device=args.device, verbose=False)
    print(f"loaded_llm_fn={model.config.llm_fn}")
    print(f"loaded_llama_backend={model.config.llama_backend}")
    model.shutdown()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
