#!/usr/bin/env -S deno run --allow-net --allow-env --allow-read --allow-write

/**
 * Google OAuth2 リフレッシュトークン取得ツール
 * 
 * 使用方法:
 * 1. .envファイルにGOOGLE_CLIENT_IDとGOOGLE_CLIENT_SECRETを設定
 * 2. このスクリプトを実行: ./tools/get-google-refresh-token.ts
 * 3. 表示されるURLをブラウザで開いて認証
 * 4. リダイレクトされたURLから認証コードをコピー
 * 5. 認証コードを入力
 * 6. 表示されたリフレッシュトークンを.envファイルに保存
 */

import { loadEnv } from "../utils/env.ts";

const OAUTH_SCOPE = [
  "https://www.googleapis.com/auth/calendar.readonly",
  "https://www.googleapis.com/auth/calendar.events.readonly",
].join(" ");

async function main() {
  console.log("=== Google OAuth2 リフレッシュトークン取得ツール ===\n");

  // .envファイルを読み込み
  await loadEnv();

  const clientId = Deno.env.get("GOOGLE_CLIENT_ID");
  const clientSecret = Deno.env.get("GOOGLE_CLIENT_SECRET");

  if (!clientId || !clientSecret) {
    console.error("エラー: 環境変数が設定されていません。");
    console.error(".envファイルに以下を設定してください:");
    console.error("  GOOGLE_CLIENT_ID=your-client-id");
    console.error("  GOOGLE_CLIENT_SECRET=your-client-secret");
    Deno.exit(1);
  }

  console.log("✅ クライアント認証情報を読み込みました\n");

  // Step 1: 認証URLを生成
  const authUrl = generateAuthUrl(clientId);
  
  console.log("📌 Step 1: 以下のURLをブラウザで開いてGoogleアカウントでログインしてください:\n");
  console.log("🔗 " + authUrl);
  console.log("\n");

  // Step 2: 認証コードを入力
  console.log("📌 Step 2: 認証後のリダイレクトURLから認証コードを取得してください。");
  console.log("リダイレクトURLは以下のような形式です:");
  console.log("http://localhost/?code=XXXXX&scope=...\n");
  
  const authCode = prompt("認証コード(code=の後の部分)を入力してください:");
  
  if (!authCode) {
    console.error("エラー: 認証コードが入力されませんでした。");
    Deno.exit(1);
  }

  // Step 3: リフレッシュトークンを取得
  console.log("\n📌 Step 3: リフレッシュトークンを取得中...");
  
  try {
    const tokens = await exchangeCodeForTokens(clientId, clientSecret, authCode);
    
    console.log("\n✅ 成功！以下の情報を.envファイルに追加してください:\n");
    console.log("=" .repeat(60));
    console.log(`GOOGLE_REFRESH_TOKEN=${tokens.refresh_token}`);
    console.log("=" .repeat(60));
    
    if (tokens.access_token) {
      console.log("\n参考情報:");
      console.log(`アクセストークン: ${tokens.access_token.substring(0, 20)}...`);
      console.log(`有効期限: ${tokens.expires_in}秒`);
    }

    // オプション: .envファイルに自動追記
    const autoSave = confirm("\n.envファイルに自動的に保存しますか？");
    if (autoSave) {
      await saveToEnvFile(tokens.refresh_token);
      console.log("✅ .envファイルに保存しました");
    }

  } catch (error) {
    console.error("\nエラー: リフレッシュトークンの取得に失敗しました");
    console.error(error);
    console.error("\n考えられる原因:");
    console.error("1. 認証コードが間違っている");
    console.error("2. 認証コードの有効期限が切れている（数分で失効）");
    console.error("3. リダイレクトURIが一致していない");
    console.error("\nもう一度最初からやり直してください。");
    Deno.exit(1);
  }
}

function generateAuthUrl(clientId: string): string {
  const params = new URLSearchParams({
    client_id: clientId,
    redirect_uri: "http://localhost",
    response_type: "code",
    scope: OAUTH_SCOPE,
    access_type: "offline",
    prompt: "consent", // 常にリフレッシュトークンを取得
  });

  return `https://accounts.google.com/o/oauth2/v2/auth?${params.toString()}`;
}

async function exchangeCodeForTokens(
  clientId: string,
  clientSecret: string,
  authCode: string
): Promise<{
  access_token: string;
  refresh_token: string;
  expires_in: number;
  token_type: string;
}> {
  const response = await fetch("https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: new URLSearchParams({
      code: authCode,
      client_id: clientId,
      client_secret: clientSecret,
      redirect_uri: "http://localhost",
      grant_type: "authorization_code",
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`トークン交換エラー: ${error}`);
  }

  const data = await response.json();
  
  if (!data.refresh_token) {
    throw new Error("リフレッシュトークンが返されませんでした。すでに認証済みの可能性があります。");
  }

  return data;
}

async function saveToEnvFile(refreshToken: string): Promise<void> {
  const envPath = ".env";
  
  try {
    // 既存の.envファイルを読み込み
    let content = "";
    try {
      content = await Deno.readTextFile(envPath);
    } catch {
      // ファイルが存在しない場合は新規作成
      console.log(".envファイルが存在しないため新規作成します");
    }

    // GOOGLE_REFRESH_TOKENが既に存在する場合は置換
    if (content.includes("GOOGLE_REFRESH_TOKEN=")) {
      content = content.replace(
        /GOOGLE_REFRESH_TOKEN=.*/,
        `GOOGLE_REFRESH_TOKEN=${refreshToken}`
      );
    } else {
      // 存在しない場合は追加
      if (content && !content.endsWith("\n")) {
        content += "\n";
      }
      content += `GOOGLE_REFRESH_TOKEN=${refreshToken}\n`;
    }

    // ファイルに書き込み
    await Deno.writeTextFile(envPath, content);
  } catch (error) {
    console.error("エラー: .envファイルの更新に失敗しました:", error);
    throw error;
  }
}

if (import.meta.main) {
  main();
}