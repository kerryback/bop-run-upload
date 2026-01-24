#!/bin/bash
# Load .env for KOYEB_API_TOKEN
if [ -f ".env" ]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        value="${value%\"}"
        value="${value#\"}"
        if [ -z "${!key}" ]; then
            export "$key=$value"
        fi
    done < .env
fi

if [ -n "$1" ]; then
    APP_NAME="$1"
else
    # Find the most recently created app
    APP_NAME=$(koyeb apps list --token "$KOYEB_API_TOKEN" -o json 2>/dev/null \
        | python -c "import json,sys; apps=[a for a in json.load(sys.stdin) if 'original' not in a.get('name','')]; apps.sort(key=lambda a: a.get('created_at','')); print(apps[-1]['name'])" 2>/dev/null)
    if [ -z "$APP_NAME" ]; then
        echo "ERROR: Could not detect most recent app."
        echo "Usage: $0 [app-name]"
        echo ""
        echo "List apps with: koyeb apps list --token \$KOYEB_API_TOKEN"
        exit 1
    fi
    echo "Using most recent app: $APP_NAME"
fi

koyeb services logs worker --app "$APP_NAME" --tail --token "$KOYEB_API_TOKEN"
