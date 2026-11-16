import argparse
import os
from pathlib import Path
import yaml
from om_ced.data.packer import prepare_splits, pull_artifacts_from_hf, push_artifacts_to_hf


def main() -> None:
    parser = argparse.ArgumentParser(description="Đóng gói hoặc đồng bộ dữ liệu .bin/.idx")
    parser.add_argument("--config", type=str, default="configs/data_prep.yaml")
    parser.add_argument("--push", action="store_true", help="Upload sau khi đóng gói xong")
    parser.add_argument("--pull", action="store_true", help="Tải sẵn từ HF Hub thay vì build lại")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    artifact_dir = Path(cfg["output"]["artifact_dir"])
    repo_id = cfg["hub"]["repo_id"]
    path_in_repo = cfg["hub"]["path_in_repo"]

    if args.pull:
        pull_artifacts_from_hf(artifact_dir, repo_id, path_in_repo)
        return

    prepare_splits(cfg)

    if args.push:
        push_artifacts_to_hf(artifact_dir, repo_id, path_in_repo)

    os._exit(0)


if __name__ == "__main__":
    main()