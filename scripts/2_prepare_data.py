import argparse
import yaml
from om_ced.data.packer import prepare_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Đóng gói dữ liệu sang .bin và .idx")
    parser.add_argument("--config", type=str, default="configs/data_prep.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prepare_splits(cfg)


if __name__ == "__main__":
    main()