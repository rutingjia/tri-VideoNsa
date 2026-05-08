#!/usr/bin/env python3
import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_json_maybe(s):
    if isinstance(s, str):
        s = s.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return s
    return s


def iter_json_file(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for x in data:
            if isinstance(x, dict):
                yield x
    elif isinstance(data, dict):
        # 常见情况：{"data": [...]} / {"annotations": [...]} / {"items": [...]}
        for key in ["data", "annotations", "items", "examples", "samples"]:
            if key in data and isinstance(data[key], list):
                for x in data[key]:
                    if isinstance(x, dict):
                        yield x
                return

        # 如果本身就是一条样本
        yield data


def iter_jsonl_file(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                x = json.loads(line)
                if isinstance(x, dict):
                    yield x
            except Exception:
                continue


def iter_parquet_file(path: Path, batch_size: int = 2048) -> Iterable[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except Exception as e:
        raise RuntimeError(
            "读取 parquet 需要 pyarrow。请先安装：pip install pyarrow pandas"
        ) from e

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        table = batch.to_pylist()
        for row in table:
            if isinstance(row, dict):
                yield row


def iter_annotation_files(ann_root: Path) -> Iterable[Path]:
    suffixes = {".json", ".jsonl", ".parquet"}

    for p in ann_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in suffixes:
            continue

        # 避免误扫 extracted 下面的东西
        parts = set(p.parts)
        if "extracted" in parts:
            continue
        yield p


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (list, dict)):
        return json.dumps(x, ensure_ascii=False)
    return str(x).strip()


def replace_video_placeholder(text: str) -> str:
    text = text.strip()

    # LLaVA 系列有些视频样本会写成 <image>，这里统一改成 Qwen3-VL/ms-swift 的 <video>
    text = text.replace("<image>", "<video>")
    text = text.replace("<video>\n", "<video>")

    if "<video>" not in text:
        text = "<video>" + text

    return text


def extract_from_conversations(sample: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    conv = None
    for key in ["conversations", "conversation", "messages"]:
        if key in sample:
            conv = load_json_maybe(sample[key])
            break

    if not isinstance(conv, list):
        return None

    # 情况 A：LLaVA 格式 [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}]
    human_text = None
    gpt_text = None

    for msg in conv:
        if not isinstance(msg, dict):
            continue

        role = normalize_text(msg.get("from", msg.get("role", ""))).lower()
        content = msg.get("value", msg.get("content", msg.get("text", "")))
        content = normalize_text(content)

        if role in ["human", "user"] and human_text is None:
            human_text = content
        elif role in ["gpt", "assistant"] and gpt_text is None:
            gpt_text = content

    if human_text and gpt_text:
        return human_text, gpt_text

    return None


def extract_from_qa_fields(sample: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    q_keys = ["question", "query", "prompt", "instruction", "input"]
    a_keys = ["answer", "response", "output", "label", "target"]

    q = None
    a = None

    for k in q_keys:
        if k in sample and sample[k] is not None:
            q = normalize_text(sample[k])
            break

    for k in a_keys:
        if k in sample and sample[k] is not None:
            a = normalize_text(sample[k])
            break

    # 多选题选项拼进 question
    options = sample.get("options", sample.get("choices", None))
    if q and options:
        options = load_json_maybe(options)
        if isinstance(options, list):
            opt_text = "\n".join([f"{chr(65+i)}. {normalize_text(o)}" for i, o in enumerate(options)])
            q = q + "\nOptions:\n" + opt_text
        elif isinstance(options, dict):
            opt_text = "\n".join([f"{k}. {normalize_text(v)}" for k, v in options.items()])
            q = q + "\nOptions:\n" + opt_text

    if q and a:
        return q, a

    return None


def get_video_field(sample: Dict[str, Any]) -> Optional[str]:
    keys = [
        "video",
        "videos",
        "video_path",
        "video_file",
        "video_name",
        "filename",
        "file_name",
        "path",
    ]

    for k in keys:
        if k not in sample:
            continue

        v = load_json_maybe(sample[k])

        if isinstance(v, list) and v:
            v = v[0]

        if isinstance(v, dict):
            for kk in ["video", "path", "file", "filename"]:
                if kk in v:
                    v = v[kk]
                    break

        if isinstance(v, str) and v.strip():
            return v.strip()

    return None


def resolve_video_path(video_ref: str, video_root: Path, basename_index: Optional[Dict[str, Path]] = None) -> Optional[str]:
    video_ref = video_ref.replace("file://", "").strip()

    candidates = []

    p = Path(video_ref)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(video_root / video_ref)

        # 有些标注里可能多一层 videos/ 或 LLaVA-Video-178K/
        candidates.append(video_root / video_ref.lstrip("./"))
        candidates.append(video_root / Path(video_ref).name)

    for c in candidates:
        if c.exists() and c.is_file():
            return str(c)

    if basename_index is not None:
        base = Path(video_ref).name
        if base in basename_index:
            return str(basename_index[base])

    return None


def build_basename_index(video_root: Path) -> Dict[str, Path]:
    print(f"[INFO] Building mp4 basename index under: {video_root}")
    index = {}
    for p in video_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            index.setdefault(p.name, p)
    print(f"[INFO] Indexed {len(index)} video files.")
    return index


def convert_one(sample: Dict[str, Any], video_root: Path, basename_index: Optional[Dict[str, Path]]) -> Optional[Dict[str, Any]]:
    video_ref = get_video_field(sample)
    if not video_ref:
        return None

    video_path = resolve_video_path(video_ref, video_root, basename_index)
    if not video_path:
        return None

    qa = extract_from_conversations(sample)
    if qa is None:
        qa = extract_from_qa_fields(sample)

    if qa is None:
        return None

    q, a = qa
    q = replace_video_placeholder(q)
    a = normalize_text(a)

    if not q or not a:
        return None

    return {
        "messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ],
        "videos": [video_path],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_root", type=str, required=True, help="LLaVA-Video-178K 标注文件根目录")
    parser.add_argument("--video_root", type=str, required=True, help="解压后的 mp4 根目录，例如 .../LLaVA-Video-178K/extracted")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=20000, help="先小规模跑通，默认取 2 万条")
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--build_index", action="store_true", help="如果路径匹配不上，扫描全部 mp4 建 basename 索引")
    args = parser.parse_args()

    ann_root = Path(args.ann_root)
    video_root = Path(args.video_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    basename_index = build_basename_index(video_root) if args.build_index else None

    files = list(iter_annotation_files(ann_root))
    print(f"[INFO] Found {len(files)} annotation files under {ann_root}")

    random.seed(args.seed)
    random.shuffle(files)

    samples = []
    total_seen = 0
    total_converted = 0
    total_missing_video = 0

    for f in files:
        print(f"[INFO] Reading {f}")

        try:
            if f.suffix.lower() == ".json":
                rows = iter_json_file(f)
            elif f.suffix.lower() == ".jsonl":
                rows = iter_jsonl_file(f)
            elif f.suffix.lower() == ".parquet":
                rows = iter_parquet_file(f)
            else:
                continue

            for row in rows:
                total_seen += 1

                item = convert_one(row, video_root, basename_index)
                if item is None:
                    # 粗略统计一下是否是视频路径问题
                    if get_video_field(row):
                        total_missing_video += 1
                    continue

                samples.append(item)
                total_converted += 1

                if args.max_samples > 0 and len(samples) >= args.max_samples:
                    break

        except Exception as e:
            print(f"[WARN] Skip file {f}: {e}")

        if args.max_samples > 0 and len(samples) >= args.max_samples:
            break

    if not samples:
        print("[ERROR] No samples converted.")
        print("[HINT] 请检查：")
        print("  1. --ann_root 下是否真的有 json/jsonl/parquet 标注")
        print("  2. --video_root 是否指向 extracted 目录")
        print("  3. 如果视频路径对不上，加 --build_index 再跑一次")
        raise SystemExit(1)

    random.shuffle(samples)

    val_n = max(1, int(len(samples) * args.val_ratio))
    val = samples[:val_n]
    train = samples[val_n:]

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for x in train:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with val_path.open("w", encoding="utf-8") as f:
        for x in val:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print("\n[DONE]")
    print(f"Total seen: {total_seen}")
    print(f"Total converted: {total_converted}")
    print(f"Potential missing-video rows: {total_missing_video}")
    print(f"Train: {len(train)} -> {train_path}")
    print(f"Val:   {len(val)} -> {val_path}")
    print("\n[Preview]")
    print(json.dumps(samples[0], ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
