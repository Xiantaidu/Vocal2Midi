import argparse
import json
from pathlib import Path

from .common import chunked
from .runtime import RomajiASROnnxModel


def load_manifest(manifest_path: str) -> list[dict]:
    items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser(description="Run romaji ASR ONNX inference with DirectML or CPU.")
    parser.add_argument("--model", default=".", help="Path to model directory or model.onnx")
    parser.add_argument("--audio", type=str, default=None, help="Single audio file")
    parser.add_argument("--manifest", type=str, default=None, help="Manifest jsonl")
    parser.add_argument("--provider", choices=["dml", "cpu"], default="dml")
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    if not args.audio and not args.manifest:
        raise ValueError("Provide either --audio or --manifest.")

    model = RomajiASROnnxModel.from_model_path(
        Path(args.model),
        provider=args.provider,
        device=args.provider,
        verbose=True,
    )

    print(f"ONNX Runtime providers: {model.session.get_providers()}")

    if args.audio:
        pred = model.transcribe([args.audio], batch_size=1)[0]
        print(f"Audio: {args.audio}")
        print(f"Predicted: {pred['text']}")

    if args.manifest:
        items = load_manifest(args.manifest)
        for batch_items in chunked(items, args.batch_size):
            preds = model.transcribe([item["audio"] for item in batch_items], batch_size=len(batch_items))
            for item, pred in zip(batch_items, preds):
                print(f"Audio: {item['audio']}")
                if "phones" in item:
                    print(f"Reference: {' '.join(item['phones'])}")
                print(f"Predicted: {pred['text']}")
                print("-" * 50)


if __name__ == "__main__":
    main()
