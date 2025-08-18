import { HubSpotConfig } from "../types/index.ts";

export class HubSpotAuth {
  private config: HubSpotConfig;

  constructor(config: HubSpotConfig) {
    this.config = config;
  }

  getApiKey(): string {
    return this.config.apiKey;
  }

  getAuthHeader(): Record<string, string> {
    return {
      "Authorization": `Bearer ${this.config.apiKey}`,
      "Content-Type": "application/json",
    };
  }

  static fromEnv(): HubSpotAuth {
    const apiKey = Deno.env.get("HUBSPOT_API_KEY");

    if (!apiKey) {
      throw new Error(
        "HubSpot API の認証情報が設定されていません。\n" +
        ".envファイルに環境変数 HUBSPOT_API_KEY を設定してください。\n\n" +
        ".env.exampleを参考に設定してください。"
      );
    }

    return new HubSpotAuth({ apiKey });
  }
}