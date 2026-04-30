import argparse
import json
from pathlib import Path

import torch
import torchaudio
from transformers import HubertForCTC
from tqdm import tqdm


def edit_distance(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c)
    return dp[n][m]


def compute_per(predictions, references):
    total_err = 0
    total_ref = 0
    for p, r in zip(predictions, references):
        total_err += edit_distance(p, r)
        total_ref += len(r)
    return 0.0 if total_ref == 0 else total_err / total_ref


def greedy_decode(logits, id2phone, blank_id=0):
    pred = torch.argmax(logits, dim=-1).tolist()
    out = []
    prev = -1
    for p in pred:
        if p != prev and p != blank_id:
            out.append(id2phone.get(p, "<unk>"))
        prev = p
    return out


def load_audio(path, target_sr=16000):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav.squeeze(0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint PER on manifest")
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--vocab", default=None, type=str)
    parser.add_argument("--sample_rate", default=16000, type=int)
    args = parser.parse_args()

    vocab_path = args.vocab
    if vocab_path is None:
        ckpt_vocab = Path(args.ckpt) / "phoneme_vocab.json"
        if ckpt_vocab.exists():
            vocab_path = str(ckpt_vocab)
        else:
            vocab_path = "data/phoneme_vocab.json"

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    id2phone = {int(v): k for k, v in vocab.items()}
    blank_id = vocab.get("<blank>", vocab.get("PAD", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HubertForCTC.from_pretrained(args.ckpt).to(device)
    model.eval()

    preds = []
    refs = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        items = [json.loads(x) for x in f]

    for item in tqdm(items, desc=f"Eval {args.ckpt}"):
        wav = load_audio(item["audio"], target_sr=args.sample_rate).to(device)
        with torch.no_grad():
            logits = model(wav.unsqueeze(0)).logits[0]
        pred = greedy_decode(logits, id2phone, blank_id=blank_id)
        ref = item["phones"]
        preds.append(pred)
        refs.append(ref)

    per = compute_per(preds, refs)
    print(f"ckpt={args.ckpt}")
    print(f"manifest={args.manifest}")
    print(f"samples={len(items)}")
    print(f"PER={per:.6f}")


if __name__ == "__main__":
    main()
