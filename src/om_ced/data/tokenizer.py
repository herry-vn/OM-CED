import itertools
import time
from pathlib import Path
from typing import Iterator

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers


def get_text_stream(hf_path: str, hf_config: str, text_col: str, skip_docs: int = 0) -> Iterator[str]:
    """Stream văn bản trực tiếp từ Hugging Face mà không tải toàn bộ dataset về đĩa."""
    ds = load_dataset(hf_path, name=hf_config, split="train", streaming=True)
    if skip_docs > 0:
        ds = ds.skip(skip_docs)
    for row in ds:
        text = row.get(text_col, "")
        if text and text.strip():
            yield text


def train_tokenizer(cfg: dict, overwrite: bool = False) -> Tokenizer:
    """Huấn luyện Byte-Level BPE tokenizer theo cấu hình YAML và lưu ra artifacts/data/."""
    artifact_dir = Path(cfg["output"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tok_path = artifact_dir / "tokenizer.json"

    if tok_path.exists() and not overwrite:
        print(f"[Tokenizer] Đã tồn tại tại {tok_path} (dùng --overwrite nếu muốn train lại).")
        return Tokenizer.from_file(str(tok_path))

    vocab_size = cfg["tokenizer"]["vocab_size"]
    if vocab_size > 65535:
        raise ValueError("vocab_size phải <= 65535 để lưu trữ bằng định dạng uint16.")

    special_tokens = cfg["tokenizer"]["special_tokens"]
    n_docs = cfg["dataset"]["tokenizer_train_docs"]

    print(f"[Tokenizer] Bắt đầu huấn luyện BPE (vocab_size={vocab_size}) trên {n_docs:,} văn bản...")
    t0 = time.time()

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=cfg["tokenizer"]["min_frequency"],
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    corpus_iter = itertools.islice(
        get_text_stream(
            cfg["dataset"]["hf_path"],
            cfg["dataset"]["hf_config"],
            cfg["dataset"]["text_column"],
        ),
        n_docs,
    )
    tokenizer.train_from_iterator(corpus_iter, trainer=trainer, length=n_docs)
    tokenizer.save(str(tok_path))

    print(f"[Tokenizer] Hoàn tất sau {time.time() - t0:.1f}s -> Đã lưu tại {tok_path}")
    return tokenizer


def load_tokenizer(cfg: dict) -> Tokenizer:
    """Tải tokenizer đã huấn luyện sẵn từ thư mục artifact_dir."""
    tok_path = Path(cfg["output"]["artifact_dir"]) / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {tok_path}. Hãy chạy 'python scripts/1_train_tokenizer.py' trước."
        )
    return Tokenizer.from_file(str(tok_path))