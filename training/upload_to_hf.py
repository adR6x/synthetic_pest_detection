"""Upload a prepared COCO dataset to Hugging Face Hub.

Run prepare_dataset.py first to create the dataset directory, then:

    python -m training.upload_to_hf --dataset_dir outputs/dataset \\
                                     --repo_id your-username/pest-detection-dataset

The repo is created automatically if it does not exist.
Log in first with:  huggingface-cli login
"""

import argparse
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def upload(dataset_dir: str, repo_id: str, private: bool, commit_message: str):
    from huggingface_hub import HfApi

    api = HfApi()

    # Create the dataset repo if it doesn't exist yet
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    print(f"Repository: https://huggingface.co/datasets/{repo_id}")

    print(f"Uploading {dataset_dir} ...")
    api.upload_folder(
        folder_path=dataset_dir,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )
    print("Upload complete.")
    print(f"\nTo train from this dataset:")
    print(f"  python -m training.train --hf_dataset {repo_id} --freeze_backbone")


def main():
    parser = argparse.ArgumentParser(description="Upload COCO dataset to Hugging Face Hub")
    parser.add_argument(
        "--dataset_dir",
        default=os.path.join(PROJECT_ROOT, "outputs", "dataset"),
        help="Path to the prepared dataset (output of prepare_dataset.py)",
    )
    parser.add_argument(
        "--repo_id",
        required=True,
        help="Hugging Face repo in the form username/dataset-name",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository (default: public)",
    )
    parser.add_argument(
        "--commit_message",
        default="Upload pest detection COCO dataset",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        raise SystemExit(
            f"Dataset directory not found: {args.dataset_dir}\n"
            "Run prepare_dataset.py first."
        )

    for required in ("images/train", "images/val", "annotations/train.json", "annotations/val.json"):
        if not os.path.exists(os.path.join(args.dataset_dir, required)):
            raise SystemExit(
                f"Expected path missing inside dataset_dir: {required}\n"
                "Make sure prepare_dataset.py completed successfully."
            )

    upload(args.dataset_dir, args.repo_id, args.private, args.commit_message)


if __name__ == "__main__":
    main()
