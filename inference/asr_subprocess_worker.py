import argparse
import json
import traceback
from pathlib import Path

import torch

from inference.asr_api import load_qwen_model


def _write_output(output_json: Path, results=None, error=None):
    payload = {
        "results": results if results is not None else [],
        "error": error,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _to_jsonable_result(item):
    """Convert model output item to a JSON-serializable structure."""
    if item is None:
        return None
    if isinstance(item, (str, int, float, bool, dict, list)):
        return item

    # Common Qwen ASR object style: has .text / .transcript
    text_attr = getattr(item, "text", None)
    if text_attr is not None:
        return {"text": str(text_attr)}

    transcript_attr = getattr(item, "transcript", None)
    if transcript_attr is not None:
        return {"transcript": str(transcript_attr)}

    # dataclass / pydantic-like objects
    if hasattr(item, "dict") and callable(getattr(item, "dict")):
        try:
            return item.dict()
        except Exception:
            pass
    if hasattr(item, "model_dump") and callable(getattr(item, "model_dump")):
        try:
            return item.model_dump()
        except Exception:
            pass

    return {"text": str(item)}


def main():
    parser = argparse.ArgumentParser(description="Qwen ASR subprocess worker")
    parser.add_argument("--input-json", required=True, type=str)
    parser.add_argument("--output-json", required=True, type=str)
    args = parser.parse_args()

    input_json = Path(args.input_json)
    output_json = Path(args.output_json)

    try:
        payload = json.loads(input_json.read_text(encoding="utf-8"))
        audio_paths = payload.get("audio_paths", [])
        model_path = payload.get("model_path")
        device = payload.get("device", "cuda")
        language = payload.get("language", "Chinese")

        if not model_path:
            raise ValueError("Missing required field: model_path")
        if not isinstance(audio_paths, list):
            raise ValueError("Field 'audio_paths' must be a list")

        model = load_qwen_model(model_path=model_path, device=device, use_cache=False)

        with torch.inference_mode():
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                with torch.amp.autocast("cuda"):
                    results = model.transcribe(audio=audio_paths, language=language)
            else:
                results = model.transcribe(audio=audio_paths, language=language)

        if not isinstance(results, list):
            results = [results]
        results = [_to_jsonable_result(x) for x in results]

        _write_output(output_json, results=results, error=None)
    except Exception:
        err = traceback.format_exc()
        _write_output(output_json, results=[], error=err)


if __name__ == "__main__":
    main()
