#!/bin/bash
# deploy_koyeb.sh - Deploy noipca workflow to Koyeb
#
# Creates a self-contained Koyeb service that:
# 1. Runs the workflow
# 2. Uploads results to S3
# 3. Auto-deletes itself to stop billing
#
# Usage:
#   ./deploy_koyeb.sh <model> <start> <end> [instance_type] [git_repo] [upload_intermediate]
#
# Arguments:
#   model: bgn, kp14, or gs21
#   start: Starting index
#   end: Ending index (exclusive)
#   instance_type: Koyeb instance type (default: 2xlarge)
#   git_repo: GitHub repo in format username/repo (default: auto-detected)
#   upload_intermediate: Upload panel/moments files to S3 (default: true)
#
# Required environment variables:
#   KOYEB_API_TOKEN: Your Koyeb API token
#   AWS_ACCESS_KEY_ID: AWS access key for S3 uploads
#   AWS_SECRET_ACCESS_KEY: AWS secret key for S3 uploads
#
# Optional environment variables:
#   S3_BUCKET: S3 bucket name (default: bop-noipca)
#   AWS_REGION: AWS region (default: us-east-2)
#   KOYEB_APP_NAME: (auto-generated from model-start-end-timestamp)
#   WORKFLOW_ID: S3 folder name (auto-generated: model-start-end-YYYYMMDD-HHMMSS)
#   KOYEB_REGION: Koyeb region (default: was)
#   MONITOR_URL: URL of koyeb-monitor service for self-termination
#
# Examples:
#   ./deploy_koyeb.sh kp14 0 10              # Uses default 2xlarge
#   ./deploy_koyeb.sh gs21 0 20              # Uses default 2xlarge
#   ./deploy_koyeb.sh bgn 0 5 4xlarge        # Override with 4xlarge

set -e  # Exit on error

# Check arguments
if [ $# -lt 3 ]; then
    echo "ERROR: Missing required arguments"
    echo ""
    echo "Usage: $0 <model> <start> <end> [instance_type] [git_repo] [upload_intermediate]"
    echo ""
    echo "Arguments:"
    echo "  model:                bgn, kp14, or gs21"
    echo "  start:                Starting index"
    echo "  end:                  Ending index (exclusive)"
    echo "  instance_type:        Koyeb instance (default: 2xlarge)"
    echo "  git_repo:             GitHub repo username/repo (default: auto-detected)"
    echo "  upload_intermediate:  Upload panel/moments to S3 (default: true)"
    echo ""
    echo "Examples:"
    echo "  $0 kp14 0 10              # Uses default 2xlarge"
    echo "  $0 gs21 0 20              # Uses default 2xlarge"
    echo "  $0 bgn 0 5 4xlarge        # Override with 4xlarge"
    exit 1
fi

# Parse arguments
MODEL=$1
START=$2
END=$3

# Validate model first (needed for default instance type)
if [[ ! "$MODEL" =~ ^(bgn|kp14|gs21)$ ]]; then
    echo "ERROR: Invalid model '$MODEL'"
    echo "Valid models: bgn, kp14, gs21"
    exit 1
fi

# Set default instance type (2xlarge for all models)
if [ -z "$4" ]; then
    INSTANCE_TYPE="2xlarge"
else
    INSTANCE_TYPE=$4
fi
GIT_REPO=${5:-kerryback/bop-run-upload}

# Validate indices
if [ "$START" -ge "$END" ]; then
    echo "ERROR: start ($START) must be less than end ($END)"
    exit 1
fi

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo "Loading credentials from .env file..."
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
    echo "Loaded credentials from .env"
fi

# Check required environment variables
if [ -z "$KOYEB_API_TOKEN" ]; then
    echo "ERROR: KOYEB_API_TOKEN environment variable not set"
    echo "Set it with: export KOYEB_API_TOKEN=your_token"
    echo "Or add it to .env file: KOYEB_API_TOKEN=your_token"
    exit 1
fi

if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "ERROR: AWS credentials not set"
    echo "Set them with:"
    echo "  export AWS_ACCESS_KEY_ID=your_key_id"
    echo "  export AWS_SECRET_ACCESS_KEY=your_secret_key"
    echo "Or add them to .env file:"
    echo "  AWS_ACCESS_KEY_ID=your_key_id"
    echo "  AWS_SECRET_ACCESS_KEY=your_secret_key"
    exit 1
fi

# Set defaults for optional variables
S3_BUCKET=${S3_BUCKET:-bop-noipca}
AWS_REGION=${AWS_REGION:-us-east-2}
KOYEB_REGION=${KOYEB_REGION:-was}
MONITOR_URL=${MONITOR_URL:-https://koyeb-monitor-kerrybackapps-c07b20b0.koyeb.app}

# Auto-detect git branch (default: current branch)
if [ -z "$GIT_BRANCH" ]; then
    GIT_BRANCH=$(git branch --show-current)
    if [ -z "$GIT_BRANCH" ]; then
        GIT_BRANCH="main"  # Fallback to main if detection fails
    fi
fi

# Generate unique app name and S3 folder name from job arguments + timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
KOYEB_APP_NAME="${MODEL}-${START}-${END}-${TIMESTAMP}"
WORKFLOW_ID="${MODEL}-${START}-${END}-${TIMESTAMP}"

echo "=========================================="
echo "KOYEB DEPLOYMENT"
echo "=========================================="
echo "Configuration:"
echo "  Model:         $MODEL"
echo "  Indices:       $START to $((END-1))"
echo "  App name:      $KOYEB_APP_NAME"
echo "  S3 folder:     $WORKFLOW_ID"
echo "  Instance type: $INSTANCE_TYPE"
echo "  Region:        $KOYEB_REGION"
echo "  Git repo:      $GIT_REPO"
echo "  Git branch:    $GIT_BRANCH"
echo "  S3 bucket:     $S3_BUCKET"
echo "  AWS region:    $AWS_REGION"
echo "=========================================="
echo ""

# Check if koyeb CLI is installed
if ! command -v koyeb &> /dev/null; then
    echo "ERROR: koyeb CLI not found"
    echo "Install it from: https://www.koyeb.com/docs/build-and-deploy/cli"
    exit 1
fi

# Check if app exists, create if it doesn't
echo "Checking if app '$KOYEB_APP_NAME' exists..."
if ! koyeb apps get "$KOYEB_APP_NAME" --token "$KOYEB_API_TOKEN" &> /dev/null; then
    echo "App doesn't exist, creating '$KOYEB_APP_NAME'..."
    koyeb apps create "$KOYEB_APP_NAME" --token "$KOYEB_API_TOKEN"
    echo "✓ App created successfully"
else
    echo "✓ App already exists"
fi

# Create the service
# The run command:
#   1. Sends init message to monitor (for tracking)
#   2. Runs main.py (which uploads logs to S3 after each step)
#   3. main.py notifies monitor at end for app termination
echo "Creating Koyeb service..."
koyeb services create worker \
  --app "$KOYEB_APP_NAME" \
  --type worker \
  --git "github.com/$GIT_REPO" \
  --git-branch "$GIT_BRANCH" \
  --git-no-deploy-on-push \
  --git-run-command "python -c \"import requests, os, datetime; requests.post(os.environ['MONITOR_URL']+'/init-logs', json={'app_name': os.environ['KOYEB_APP_NAME'], 'model': '$MODEL', 'start': $START, 'end': $END, 'instance_type': '$INSTANCE_TYPE', 'started_at': datetime.datetime.now().isoformat()})\"; python main.py $MODEL $START $END --koyeb" \
  --instance-type "$INSTANCE_TYPE" \
  --regions "$KOYEB_REGION" \
  --env MONITOR_URL="$MONITOR_URL" \
  --env S3_BUCKET="$S3_BUCKET" \
  --env AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  --env AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  --env AWS_REGION="$AWS_REGION" \
  --env KOYEB_API_TOKEN="$KOYEB_API_TOKEN" \
  --env KOYEB_APP_NAME="$KOYEB_APP_NAME" \
  --env WORKFLOW_ID="$WORKFLOW_ID" \
  --token "$KOYEB_API_TOKEN"

echo ""
echo "=========================================="
echo "APP CREATED SUCCESSFULLY"
echo "=========================================="
echo ""
echo "The service will:"
echo "  1. Build and deploy your code"
echo "  2. Initialize log file at monitor (for tracking)"
echo "  3. Run: python main.py $MODEL $START $END --koyeb"
echo "  4. Upload results and logs to s3://$S3_BUCKET/koyeb-results/$WORKFLOW_ID/"
echo "     (logs uploaded after each step, results uploaded as completed)"
echo "  5. Notify monitor at completion for app termination"
echo ""
echo "Monitor the service:"
echo "  koyeb services get worker --app $KOYEB_APP_NAME --token \$KOYEB_API_TOKEN"
echo ""
echo "View build logs:"
echo "  koyeb services logs worker --app $KOYEB_APP_NAME -t build --token \$KOYEB_API_TOKEN"
echo ""
echo "View runtime logs (live):"
echo "  koyeb services logs worker --app $KOYEB_APP_NAME --tail --token \$KOYEB_API_TOKEN"
echo ""
echo "View captured logs (after completion):"
echo "  $MONITOR_URL/logs/$KOYEB_APP_NAME"
echo ""
echo "Download results when complete:"
echo "  source .env && aws s3 ls s3://$S3_BUCKET/koyeb-results/"
echo "  source .env && aws s3 sync s3://$S3_BUCKET/koyeb-results/$WORKFLOW_ID/ ./results/"
echo ""
echo "=========================================="
