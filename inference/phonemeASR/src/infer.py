# File: src/infer.py
import argparse
import json
import torch
import torchaudio
from transformers import HubertForCTC
from pathlib import Path


def greedy_decode(logits, id2phone, blank=0):
    """CTC greedy decode: collapse repeats and remove blank."""
    pred = logits.argmax(-1).tolist()
    out = []
    prev = -1
    for p in pred:
        if p != prev and p != blank:
            out.append(id2phone.get(p, "<unk>"))
        prev = p
    return out


def main():
    parser = argparse.ArgumentParser(description="Phoneme ASR inference")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--manifest", type=str, help="Path to test manifest file")
    parser.add_argument("--audio", type=str, help="Path to single audio file for testing")
    parser.add_argument("--vocab", type=str, default=None,
                        help="Path to vocab json file (auto-detected from ckpt dir if not provided)")
    args = parser.parse_args()

    # Auto-detect vocab path
    vocab_path = args.vocab
    if vocab_path is None:
        # Try to find vocab in checkpoint directory
        ckpt_vocab = Path(args.ckpt) / "phoneme_vocab.json"
        if ckpt_vocab.exists():
            vocab_path = str(ckpt_vocab)
            print(f"Auto-detected vocab: {vocab_path}")
        else:
            # Fallback to default project location
            default_vocab = Path("data/phoneme_vocab.json")
            if default_vocab.exists():
                vocab_path = str(default_vocab)
                print(f"Using default vocab: {vocab_path}")
            else:
                print("Error: No vocab file found. Specify --vocab explicitly.")
                return

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    id2phone = {int(v): k for k, v in vocab.items()}

    # Find blank id (should be 0)
    blank_id = vocab.get("<blank>", vocab.get("PAD", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HubertForCTC.from_pretrained(args.ckpt).to(device)
    model.eval()

    def infer_single(wav_path):
        waveform, sr = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        input_values = waveform.to(device)
        with torch.no_grad():
            logits = model(input_values).logits[0]

        return greedy_decode(logits, id2phone, blank=blank_id)

    if args.audio:
        phones = infer_single(args.audio)
        print(f"Audio: {args.audio}")
        print(f"Predicted: {' '.join(phones)}")

    if args.manifest:
        with open(args.manifest, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                phones = infer_single(item["audio"])
                print(f"Audio: {item['audio']}")
                print(f"Reference: {' '.join(item['phones'])}")
                print(f"Predicted: {' '.join(phones)}")
                print("-" * 50)


if __name__ == "__main__":
    main()