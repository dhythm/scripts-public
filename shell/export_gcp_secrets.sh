#!/bin/bash
#
# Usage: ./export_gcp_secrets.sh <PROJECT_ID>
#
# This script fetches all secrets from a GCP project and creates a .env file
# with the secret names and values.

set -e  # Exit on error

# Check if PROJECT_ID is provided as argument
if [ $# -eq 0 ]; then
    echo "Error: PROJECT_ID is required"
    echo "Usage: $0 <PROJECT_ID>"
    exit 1
fi

PROJECT_ID="$1"

# Cleanup function
cleanup() {
    rm -f secrets_list.txt
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Get list of secrets
echo "Fetching secrets list from project: $PROJECT_ID"
gcloud secrets list --project="$PROJECT_ID" --format="value(name)" > secrets_list.txt

# .env を初期化（空にする）
> .env

while read SECRET_NAME; do
  # 各シークレットの最新バージョンを取得
  SECRET_VALUE=$(gcloud secrets versions access latest --secret="$SECRET_NAME" --project="$PROJECT_ID")

  # 改行を含む場合エスケープ（必要に応じて）
  SECRET_VALUE_ESCAPED=$(echo "$SECRET_VALUE" | sed ':a;N;$!ba;s/\n/\\n/g')

  # .env に追記
  echo "${SECRET_NAME}=${SECRET_VALUE_ESCAPED}" >> .env

  echo "Fetched secret: $SECRET_NAME"
done < secrets_list.txt

rm secrets_list.txt