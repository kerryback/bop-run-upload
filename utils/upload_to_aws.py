"""
AWS S3 upload utilities for incremental file uploads.

Provides functions to upload files to S3 as they are created.
Raises exceptions on failure after retry attempts.

Usage:
    from utils.upload_to_aws import upload_file, is_s3_configured

    if is_s3_configured():
        upload_file('/path/to/file.pkl')
"""

import os
import time
import boto3
from pathlib import Path
from datetime import datetime

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds between retries


def is_s3_configured():
    """
    Check if S3 environment variables are configured.

    Returns:
        bool: True if S3_BUCKET and AWS credentials are set
    """
    return bool(
        os.environ.get('S3_BUCKET') and
        os.environ.get('AWS_ACCESS_KEY_ID') and
        os.environ.get('AWS_SECRET_ACCESS_KEY')
    )


def get_s3_client():
    """
    Create and return a boto3 S3 client.

    Returns:
        boto3.client: S3 client

    Raises:
        RuntimeError: If S3 is not configured
        Exception: If client creation fails
    """
    if not is_s3_configured():
        raise RuntimeError(
            "S3 not configured. Required environment variables: "
            "S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY"
        )

    aws_region = os.environ.get('AWS_REGION', 'us-east-2')
    s3_client = boto3.client('s3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=aws_region
    )
    return s3_client


def upload_file(filepath, verbose=True, max_retries=MAX_RETRIES):
    """
    Upload a single file to S3 with retry logic.

    Retries up to max_retries times on failure. Raises on final failure.

    Args:
        filepath: Path to local file (absolute or relative)
        verbose: If True, print progress messages (default: True)
        max_retries: Number of upload attempts (default: 3)

    Raises:
        FileNotFoundError: If the file does not exist
        RuntimeError: If S3 is not configured or all upload attempts fail
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    s3_client = get_s3_client()

    bucket_name = os.environ.get('S3_BUCKET')
    workflow_id = os.environ.get('WORKFLOW_ID', datetime.now().strftime('%Y%m%d_%H%M%S'))

    # Determine S3 key based on file location
    if '/outputs/' in str(filepath) or '\\outputs\\' in str(filepath):
        s3_key = f'koyeb-results/{workflow_id}/outputs/{filepath.name}'
    elif '/logs/' in str(filepath) or '\\logs\\' in str(filepath):
        s3_key = f'koyeb-results/{workflow_id}/logs/{filepath.name}'
    else:
        s3_key = f'koyeb-results/{workflow_id}/outputs/{filepath.name}'

    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            s3_client.upload_file(str(filepath), bucket_name, s3_key)
            if verbose:
                print(f"  Uploaded {filepath.name} to s3://{bucket_name}/{s3_key}")
            return
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                print(f"  Upload attempt {attempt}/{max_retries} failed for {filepath.name}: {e}")
                print(f"  Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  Upload attempt {attempt}/{max_retries} failed for {filepath.name}: {e}")

    raise RuntimeError(
        f"S3 upload failed after {max_retries} attempts for {filepath.name}: {last_exception}"
    )
