#!/bin/bash
# list_s3.sh - List folders and files in the AWS S3 bucket
#
# Usage:
#   ./list_s3.sh [path]
#
# Arguments:
#   path: Optional S3 path within the bucket (default: lists root)
#
# Required environment variables (can be set in .env file):
#   AWS_ACCESS_KEY_ID: AWS access key
#   AWS_SECRET_ACCESS_KEY: AWS secret key
#
# Optional environment variables:
#   S3_BUCKET: S3 bucket name (default: bop-noipca)
#   AWS_REGION: AWS region (default: us-east-2)
#
# Examples:
#   ./list_s3.sh                      # List root of bucket
#   ./list_s3.sh koyeb-results/       # List koyeb-results folder
#   ./list_s3.sh koyeb-results/20250125-123456/  # List specific run

set -e

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue

        # Remove leading/trailing whitespace and quotes
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        value="${value%\"}"
        value="${value#\"}"

        # Only set if not already set (explicit env vars take priority)
        if [ -z "${!key}" ]; then
            export "$key=$value"
        fi
    done < .env
fi

# Check required environment variables
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "ERROR: AWS credentials not set"
    echo "Set them with:"
    echo "  export AWS_ACCESS_KEY_ID=your_key_id"
    echo "  export AWS_SECRET_ACCESS_KEY=your_secret_key"
    echo "Or add them to .env file"
    exit 1
fi

# Set defaults
S3_BUCKET=${S3_BUCKET:-bop-noipca}
AWS_REGION=${AWS_REGION:-us-east-2}

# Parse optional path argument
S3_PATH=${1:-}

# Export for AWS CLI
export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION=$AWS_REGION

echo "Listing s3://$S3_BUCKET/$S3_PATH"
echo "=========================================="
aws s3 ls "s3://$S3_BUCKET/$S3_PATH"
