#!/bin/bash

PROJECT_ID=<PROJECT_ID>

gcloud secrets list --project=<PROJECT_ID> --format="value(name)" > secrets_list.txt

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