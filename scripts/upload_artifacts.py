"""Upload everything in inference/artifacts to S3. Config comes from .env."""

import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

ARTIFACTS_DIR = Path("artifacts")
load_dotenv()

bucket = os.environ["S3_BUCKET_NAME"]
prefix = os.environ.get("S3_INFERENCE_ARTIFACTS_PREFIX", "").strip("/")
region = os.environ.get("AWS_DEFAULT_REGION")

s3 = boto3.client("s3", region_name=region)
files = sorted(p for p in ARTIFACTS_DIR.iterdir() if p.is_file())

if not files:
    raise SystemExit(f"No files found in {ARTIFACTS_DIR}")

print(f"Uploading {len(files)} file(s) to s3://{bucket}/{prefix}/")
for path in files:
    key = f"{prefix}/{path.name}" if prefix else path.name
    print(f"  {path.name}")
    s3.upload_file(str(path), bucket, key)

print("Done.")
