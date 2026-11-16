import argparse
import yaml
from om_ced.data.tokenizer import train_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện BPE Tokenizer")
    parser.add_argument("--config", type=str, default="configs/data_prep.yaml")
    parser.add_argument("--overwrite", action="store_true", help="Train đè nếu tokenizer.json đã tồn tại")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_tokenizer(cfg, overwrite=args.overwrite)


if __name__ == "__main__":
    main()