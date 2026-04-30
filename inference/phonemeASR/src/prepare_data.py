# File: src/prepare_data.py
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
import textgrid as tg

PHONE_TIER_NAME = "phones"

# 所有需要折叠为 "sil" 的静音/呼吸符号（包含大小写变体）
SIL_SET = {"sp", "SP", "pau", "sil", "", "br", "BR", "Sil", "Pau"}

# 音素归一化映射：将 TextGrid 中出现的变体统一到 checkpoint vocab 的标准形式
# checkpoint vocab: PAD=0, UNK=1, SOS=2, EOS=3, a=4, i=5, u=6, e=7, o=8,
#   I=9, U=10, k=11, g=12, s=13, z=14, t=15, d=16, n=17, h=18, b=19, p=20,
#   m=21, y=22, r=23, w=24, f=25, j=26, v=27, N=28, cl=29, sh=30, ch=31,
#   ts=32, ky=33, gy=34, hy=35, by=36, py=37, my=38, ny=39, ry=40, fy=41,
#   ty=42, dy=43, kw=44, gw=45, pau=46, sil=47
PHONE_NORM_MAP = {
    # 促音变体
    "q": "cl",
    "Q": "cl",
    # 大写元音（保留，checkpoint 有 I=9, U=10）
    # 小写元音（保留，checkpoint 有 a=4, i=5, u=6, e=7, o=8）
    # 其余直接保持原样，checkpoint vocab 覆盖了绝大部分日语音素
}

# checkpoint 支持的有效音素集合（不含特殊 token PAD/UNK/SOS/EOS/pau/sil）
CHECKPOINT_PHONEMES = {
    "a", "i", "u", "e", "o", "I", "U",
    "k", "g", "s", "z", "t", "d", "n", "h", "b", "p", "m", "y", "r", "w", "f", "j", "v",
    "N", "cl",
    "sh", "ch", "ts",
    "ky", "gy", "hy", "by", "py", "my", "ny", "ry", "fy", "ty", "dy",
    "kw", "gw",
}


def normalize_phone(ph: str) -> str:
    """将单个音素归一化为 checkpoint vocab 中的标准形式。"""
    # 先查映射表
    if ph in PHONE_NORM_MAP:
        return PHONE_NORM_MAP[ph]
    return ph


def extract_phones(tg_path: str) -> list[str]:
    """从 TextGrid 中提取归一化后的音素序列。"""
    g = tg.TextGrid.fromFile(tg_path)
    tier = g.getFirst(PHONE_TIER_NAME)
    seq = []
    for interval in tier:
        ph = interval.mark.strip()
        if ph in SIL_SET:
            ph = "sil"
        if not ph:
            continue
        # 归一化
        ph = normalize_phone(ph)
        # 折叠连续 sil
        if ph == "sil" and seq and seq[-1] == "sil":
            continue
        seq.append(ph)
    # 去掉首尾 sil
    while seq and seq[0] == "sil":
        seq.pop(0)
    while seq and seq[-1] == "sil":
        seq.pop()
    return seq


def _get_song_name(wav_path: Path) -> str:
    """从文件名中提取歌曲名（去掉末尾的 _NNN 编号）。
    
    例如: 'Lemon - 米津玄師_Vocals_noreverb_003.wav' -> 'Lemon - 米津玄師_Vocals_noreverb'
    """
    stem = wav_path.stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def build_manifest(raw_dir: str, out_dir: str, dev_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
    """解析所有 wav+TextGrid，按歌曲级别划分 train/dev/test 并写入 jsonl。"""
    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 按歌曲分组
    song_items = defaultdict(list)
    for wav in sorted(raw.glob("*.wav")):
        tgp = wav.with_suffix(".TextGrid")
        if not tgp.exists():
            continue
        phones = extract_phones(str(tgp))
        if len(phones) == 0:
            continue
        song = _get_song_name(wav)
        song_items[song].append({
            "audio": str(wav.resolve()),
            "phones": phones,
            "n_phones": len(phones),
        })

    songs = sorted(song_items.keys())
    print(f"Found {len(songs)} songs, {sum(len(v) for v in song_items.values())} total clips")

    if len(songs) < 3:
        # 歌曲太少，全部放 train，复制一份作 dev
        print("Warning: fewer than 3 songs, using all for train and duplicating as dev")
        train_songs = songs
        dev_songs = songs
        test_songs = []
    else:
        # 歌曲级别随机划分
        random.seed(seed)
        shuffled = songs[:]
        random.shuffle(shuffled)

        n_dev = max(1, int(len(shuffled) * dev_ratio))
        n_test = max(1, int(len(shuffled) * test_ratio))
        n_train = len(shuffled) - n_dev - n_test

        if n_train < 1:
            n_train = 1
            n_dev = max(1, (len(shuffled) - 1) // 2)
            n_test = len(shuffled) - 1 - n_dev

        train_songs = shuffled[:n_train]
        dev_songs = shuffled[n_train:n_train + n_dev]
        test_songs = shuffled[n_train + n_dev:]

    def _write_split(split_name, split_songs):
        items = []
        for s in split_songs:
            items.extend(song_items[s])
        path = out / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(items)} clips from {len(split_songs)} songs -> {path}")
        return path

    print("Song-level split:")
    print(f"  train songs: {train_songs}")
    print(f"  dev songs:   {dev_songs}")
    print(f"  test songs:  {test_songs}")

    train_path = _write_split("train", train_songs)
    dev_path = _write_split("dev", dev_songs)
    if test_songs:
        _write_split("test", test_songs)

    return str(train_path), str(dev_path)


def build_vocab_from_checkpoint(ckpt_vocab_path: str, out_path: str):
    """基于 checkpoint 的 vocab 构建训练用 phoneme_vocab.json。
    
    策略 A：将 checkpoint vocab 的映射关系重新整理为训练所需格式。
    - <blank> 对应 PAD (index 0)
    - <unk> 对应 UNK (index 1)
    - 其余音素保持 checkpoint 的索引
    - 移除 SOS/EOS（CTC 不需要）
    """
    with open(ckpt_vocab_path, "r", encoding="utf-8") as f:
        ckpt_vocab = json.load(f)

    # 直接使用 checkpoint vocab，但重命名特殊 token 以匹配训练代码的约定
    vocab = {}
    for phone, idx in ckpt_vocab.items():
        if phone == "PAD":
            vocab["<blank>"] = idx
        elif phone == "UNK":
            vocab["<unk>"] = idx
        elif phone in ("SOS", "EOS"):
            # CTC 训练不需要 SOS/EOS，但保留其索引位置避免错位
            # 训练代码中不会使用这些 token，但保留它们可以让 vocab_size 与 checkpoint 一致
            vocab[phone] = idx
        else:
            vocab[phone] = idx

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Built vocab (size {len(vocab)}) from checkpoint -> {out_path}")
    return vocab


def build_vocab_from_data(manifest_paths: list[str], out_path: str):
    """从 manifest 中扫描所有出现的音素，自动构建 vocab（策略 B 备选）。"""
    vocab = {"<blank>": 0, "<unk>": 1}
    for mp in manifest_paths:
        with open(mp, encoding="utf-8") as f:
            for line in f:
                for p in json.loads(line)["phones"]:
                    if p not in vocab:
                        vocab[p] = len(vocab)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Built vocab from data (size {len(vocab)}) -> {out_path}")
    return vocab


def validate_vocab_coverage(manifest_path: str, vocab: dict):
    """检查 manifest 中所有音素是否都在 vocab 中，输出未覆盖的音素。"""
    missing = set()
    total = 0
    unk_count = 0
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            for p in json.loads(line)["phones"]:
                total += 1
                if p not in vocab:
                    missing.add(p)
                    unk_count += 1
    if missing:
        print(f"WARNING: {len(missing)} unseen phones ({unk_count}/{total} tokens): {sorted(missing)}")
    else:
        print(f"OK: all {total} phone tokens covered by vocab")
    return missing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for phoneme ASR training")
    parser.add_argument("--raw", type=str, help="Raw data directory containing wav and TextGrid files")
    parser.add_argument("--out_dir", type=str, help="Output directory for manifest files")
    parser.add_argument("--ckpt_vocab", type=str, default=None,
                        help="Path to checkpoint vocab.json for Strategy A vocab building")
    parser.add_argument("--vocab_out", type=str, default=None,
                        help="Output path for phoneme_vocab.json")
    parser.add_argument("--inspect", type=str, help="Inspect a manifest file (print first 5 lines)")
    parser.add_argument("--dev_ratio", type=float, default=0.1, help="Dev set ratio (by song count)")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio (by song count)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    if args.inspect:
        with open(args.inspect, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(line.strip())

    elif args.raw and args.out_dir:
        # Step 1: Build manifests with song-level split
        train_path, dev_path = build_manifest(
            args.raw, args.out_dir,
            dev_ratio=args.dev_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

        # Step 2: Build vocab
        vocab_out = args.vocab_out or str(Path(args.out_dir).parent / "phoneme_vocab.json")
        if args.ckpt_vocab:
            # Strategy A: align to checkpoint vocab
            vocab = build_vocab_from_checkpoint(args.ckpt_vocab, vocab_out)
        else:
            # Strategy B: build from data
            manifest_paths = [train_path]
            # Include dev for vocab coverage (dev songs may have rare phones)
            if Path(dev_path).exists():
                manifest_paths.append(dev_path)
            test_path = str(Path(args.out_dir) / "test.jsonl")
            if Path(test_path).exists():
                manifest_paths.append(test_path)
            vocab = build_vocab_from_data(manifest_paths, vocab_out)

        # Step 3: Validate coverage
        print("\nValidating vocab coverage:")
        validate_vocab_coverage(train_path, vocab)
        if Path(dev_path).exists():
            validate_vocab_coverage(dev_path, vocab)
        test_path = str(Path(args.out_dir) / "test.jsonl")
        if Path(test_path).exists():
            validate_vocab_coverage(test_path, vocab)

    else:
        parser.print_help()