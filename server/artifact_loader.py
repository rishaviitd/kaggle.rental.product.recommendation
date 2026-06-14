import os
from pathlib import Path

import boto3


ARTIFACTS_DIR = Path("artifacts/final")


def download_artifacts() -> Path:
    bucket = os.environ["S3_BUCKET_NAME"]
    prefix = os.environ.get(
        "S3_INFERENCE_ARTIFACTS_PREFIX",
        "",
    ).strip("/")

    print(f"Downloading model artifacts from s3://{bucket}/{prefix}/")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client(
        "s3",
        region_name=os.environ.get("AWS_DEFAULT_REGION"),
    )
    paginator = s3.get_paginator("list_objects_v2")
    downloaded = 0
    object_prefix = f"{prefix}/" if prefix else ""

    for page in paginator.paginate(Bucket=bucket, Prefix=object_prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/"):
                continue

            relative_name = key.removeprefix(object_prefix)
            destination = ARTIFACTS_DIR / relative_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(destination))
            downloaded += 1

    if downloaded == 0:
        raise RuntimeError(
            f"No artifacts found at s3://{bucket}/{prefix}/"
        )

    print(f"Downloaded {downloaded} model artifacts.")
    return ARTIFACTS_DIR
