import itertools
from pathlib import Path
from typing import Iterator, List

import numpy as np
import yaml
from tokenizers import Tokenizer
from tqdm import tqdm

from om_ced.data.tokenizer import get_text_stream, load_tokenizer


def write_split_to_bin_idx(
    split_name: str,
    text_iter: Iterator[str],
    tokenizer: Tokenizer,
    target_tokens: int,
    batch_docs: int,
    artifact_dir: Path,
) -> dict:
    """
    Tokenize theo lô và ghi cuốn chiếu ra 2 file:
      - {split}.bin: Mảng phẳng uint16 chứa toàn bộ token IDs nối tiếp nhau.
      - {split}.idx: Mảng uint64 chứa offset bắt đầu/kết thúc của từng văn bản.
    """
    bin_path = artifact_dir / f"{split_name}.bin"
    idx_path = artifact_dir / f"{split_name}.idx"

    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    if bos_id is None or eos_id is None:
        raise ValueError("Tokenizer thiếu token đặc biệt <|bos|> hoặc <|eos|>.")

    offsets: List[int] = [0]
    total_tokens = 0
    total_docs = 0

    pbar = tqdm(total=target_tokens, unit="tok", unit_scale=True, desc=f"Build {split_name}")

    with open(bin_path, "wb") as f_bin:
        while total_tokens < target_tokens:
            batch_texts = list(itertools.islice(text_iter, batch_docs))
            if not batch_texts:
                break

            encodings = tokenizer.encode_batch(batch_texts)
            chunk_tokens: List[int] = []

            for enc in encodings:
                doc_ids = [bos_id] + enc.ids + [eos_id]
                chunk_tokens.extend(doc_ids)

                total_tokens += len(doc_ids)
                total_docs += 1
                offsets.append(total_tokens)

                if total_tokens >= target_tokens:
                    break

            if chunk_tokens:
                arr = np.asarray(chunk_tokens, dtype=np.uint16)
                f_bin.write(arr.tobytes())
                pbar.update(len(chunk_tokens))

    pbar.close()

    idx_arr = np.asarray(offsets, dtype=np.uint64)
    idx_arr.tofile(idx_path)

    bin_size_mb = bin_path.stat().st_size / (1024 * 1024)
    print(
        f"[{split_name.upper()}] {total_tokens:,} tokens | {total_docs:,} docs | "
        f"{bin_path.name}: {bin_size_mb:.2f} MB | {idx_path.name}: {len(idx_arr):,} offsets"
    )
    return {"tokens": int(total_tokens), "docs": int(total_docs)}


def run_sanity_check(artifact_dir: Path, tokenizer: Tokenizer) -> None:
    """Đọc ngược lại file .bin và .idx bằng memmap để kiểm tra tính toàn vẹn dữ liệu."""
    val_bin = np.memmap(artifact_dir / "val.bin", dtype=np.uint16, mode="r")
    val_idx = np.memmap(artifact_dir / "val.idx", dtype=np.uint64, mode="r")

    if len(val_bin) != val_idx[-1]:
        raise AssertionError("Lệch kích thước giữa val.bin và offset cuối cùng trong val.idx!")

    start, end = int(val_idx[0]), int(val_idx[1])
    sample_ids = val_bin[start:end].tolist()
    decoded_text = tokenizer.decode(sample_ids, skip_special_tokens=False)

    print("\n--- SANITY CHECK (Đọc ngược từ val.bin + val.idx) ---")
    print(f"Tổng số văn bản trong val: {len(val_idx) - 1:,}")
    print(f"Độ dài văn bản #0: {end - start} tokens | ID đầu: {sample_ids[0]}, ID cuối: {sample_ids[-1]}")
    print(f"Nội dung giải mã (250 ký tự đầu):\n{decoded_text[:250]!r}")
    print("-----------------------------------------------------\n")


def prepare_splits(cfg: dict) -> None:
    """Điều phối tạo val.bin/.idx, train.bin/.idx và xuất file meta.yaml."""
    artifact_dir = Path(cfg["output"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(cfg)

    # Bỏ qua các văn bản đã dùng để train tokenizer nhằm tránh rò rỉ dữ liệu
    skip_docs = cfg["dataset"]["tokenizer_train_docs"]
    stream = get_text_stream(
        cfg["dataset"]["hf_path"],
        cfg["dataset"]["hf_config"],
        cfg["dataset"]["text_column"],
        skip_docs=skip_docs,
    )

    batch_docs = cfg["dataset"]["batch_docs"]
    val_stats = write_split_to_bin_idx(
        split_name="val",
        text_iter=stream,
        tokenizer=tokenizer,
        target_tokens=cfg["dataset"]["val_target_tokens"],
        batch_docs=batch_docs,
        artifact_dir=artifact_dir,
    )
    train_stats = write_split_to_bin_idx(
        split_name="train",
        text_iter=stream,
        tokenizer=tokenizer,
        target_tokens=cfg["dataset"]["train_target_tokens"],
        batch_docs=batch_docs,
        artifact_dir=artifact_dir,
    )

    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "pad_token_id": tokenizer.token_to_id("<|pad|>"),
        "bos_token_id": tokenizer.token_to_id("<|bos|>"),
        "eos_token_id": tokenizer.token_to_id("<|eos|>"),
        "unk_token_id": tokenizer.token_to_id("<|unk|>"),
        "dtype": "uint16",
        "splits": {"val": val_stats, "train": train_stats},
    }
    with open(artifact_dir / "meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    run_sanity_check(artifact_dir, tokenizer)