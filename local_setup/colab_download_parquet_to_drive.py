from pathlib import Path
from huggingface_hub import HfApi, snapshot_download


REPO_ID = "medarc/nanopath"
DRIVE_DATASET_DIR = "/content/drive/MyDrive/nanopath_parquet"
MAX_WORKERS = 4


def main():
    dataset_dir = Path(DRIVE_DATASET_DIR)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    all_files = HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    all_shards = sorted(path for path in all_files if path.startswith("shard-") and path.endswith(".parquet"))
    target_shards = all_shards
    present_shards = {path.name for path in dataset_dir.glob("shard-*.parquet") if path.stat().st_size > 0}
    missing_shards = [name for name in target_shards if name not in present_shards]

    print(f"total shards on hub: {len(all_shards)}")
    print(f"target shards this run: {len(target_shards)}")
    print(f"already present: {len(present_shards.intersection(target_shards))}")
    print(f"missing: {len(missing_shards)}")
    print(f"destination: {dataset_dir}")

    if missing_shards:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(dataset_dir),
            allow_patterns=missing_shards,
            max_workers=MAX_WORKERS,
        )
        print(f"downloaded {len(missing_shards)} shard(s)")
    else:
        print("nothing to download")

if __name__ == "__main__":
    main()