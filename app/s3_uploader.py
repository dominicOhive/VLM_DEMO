import os

import boto3


AWS_REGION = (
    os.getenv("AWS_REGION")
    or os.getenv("AWS_DEFAULT_REGION")
    or "ap-northeast-2"
)
S3_BUCKET = os.getenv("AWS_S3_DEV_BUCKET_NAME") or os.getenv("S3_BUCKET_NAME")
S3_PUBLIC_BASE_URL = os.getenv("S3_URL")


def upload_file_to_s3(path: str, key: str, content_type: str = "image/jpeg") -> str:
    if not S3_BUCKET:
        raise ValueError("AWS_S3_DEV_BUCKET_NAME or S3_BUCKET_NAME must be set")

    client = boto3.client("s3", region_name=AWS_REGION)
    client.upload_file(
        path,
        S3_BUCKET,
        key,
        ExtraArgs={"ContentType": content_type},
    )
    return _public_url(key)


def _public_url(key: str) -> str:
    if S3_PUBLIC_BASE_URL:
        return f"{S3_PUBLIC_BASE_URL.rstrip('/')}/{key}"
    return f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
