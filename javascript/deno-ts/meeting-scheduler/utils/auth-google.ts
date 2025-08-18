import { GoogleCalendarConfig } from "../types/index.ts";

export class GoogleAuth {
  private config: GoogleCalendarConfig;
  private accessToken?: string;
  private tokenExpiry?: Date;

  constructor(config: GoogleCalendarConfig) {
    this.config = config;
  }

  async getAccessToken(): Promise<string> {
    if (this.accessToken && this.tokenExpiry && this.tokenExpiry > new Date()) {
      return this.accessToken;
    }

    const tokenUrl = "https://oauth2.googleapis.com/token";
    const params = new URLSearchParams({
      client_id: this.config.clientId,
      client_secret: this.config.clientSecret,
      refresh_token: this.config.refreshToken,
      grant_type: "refresh_token",
    });

    const response = await fetch(tokenUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: params.toString(),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Google OAuth2 トークン取得エラー: ${error}`);
    }

    const data = await response.json();
    this.accessToken = data.access_token;
    this.tokenExpiry = new Date(Date.now() + (data.expires_in - 60) * 1000);

    return this.accessToken;
  }

  static fromEnv(): GoogleAuth {
    const clientId = Deno.env.get("GOOGLE_CLIENT_ID");
    const clientSecret = Deno.env.get("GOOGLE_CLIENT_SECRET");
    const refreshToken = Deno.env.get("GOOGLE_REFRESH_TOKEN");

    if (!clientId || !clientSecret || !refreshToken) {
      throw new Error(
        "Google Calendar API の認証情報が設定されていません。\n" +
        ".envファイルに以下の環境変数を設定してください:\n" +
        "- GOOGLE_CLIENT_ID\n" +
        "- GOOGLE_CLIENT_SECRET\n" +
        "- GOOGLE_REFRESH_TOKEN\n\n" +
        ".env.exampleを参考に設定してください。"
      );
    }

    return new GoogleAuth({
      clientId,
      clientSecret,
      refreshToken,
    });
  }
}