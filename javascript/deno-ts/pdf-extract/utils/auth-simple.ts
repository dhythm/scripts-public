import { ServiceAccountKey } from "./auth.ts";
import { GoogleAuthHelper } from "./auth.ts";

// シンプルなGoogle Cloud API認証ヘルパー
export class SimpleGoogleAuth {
  private credentials?: ServiceAccountKey;
  private projectId?: string;
  private authHelper?: GoogleAuthHelper;

  async loadCredentials(): Promise<void> {
    const credentialsPath = Deno.env.get("GOOGLE_APPLICATION_CREDENTIALS");
    
    if (!credentialsPath) {
      throw new Error(
        "環境変数 GOOGLE_APPLICATION_CREDENTIALS が設定されていません。\n" +
        "Google Cloud サービスアカウントキーのパスを設定してください。"
      );
    }

    const keyFile = await Deno.readTextFile(credentialsPath);
    this.credentials = JSON.parse(keyFile) as ServiceAccountKey;
    
    // サービスアカウントキーからproject_idを自動的に取得
    if (this.credentials.project_id) {
      this.projectId = this.credentials.project_id;
    } else {
      // フォールバック: 環境変数から取得
      this.projectId = Deno.env.get("GOOGLE_CLOUD_PROJECT");
      if (!this.projectId) {
        throw new Error(
          "プロジェクトIDが見つかりません。\n" +
          "サービスアカウントキーにproject_idが含まれているか、\n" +
          "環境変数 GOOGLE_CLOUD_PROJECT を設定してください。"
        );
      }
    }

    // GoogleAuthHelperを初期化
    this.authHelper = new GoogleAuthHelper();
    await this.authHelper.authenticate();
  }

  getProjectId(): string {
    if (!this.projectId) {
      throw new Error("プロジェクトIDが設定されていません。");
    }
    return this.projectId;
  }

  getApiKey(): string | undefined {
    return Deno.env.get("GOOGLE_API_KEY");
  }

  // Google CloudのAPIキーを使用した簡易認証
  async getAuthHeaders(): Promise<Record<string, string>> {
    const apiKey = this.getApiKey();
    
    if (apiKey) {
      return {
        "x-goog-api-key": apiKey,
      };
    }

    // サービスアカウントからアクセストークンを取得
    if (this.authHelper) {
      try {
        const accessToken = await this.authHelper.getAccessToken();
        return {
          "Authorization": `Bearer ${accessToken}`,
        };
      } catch (error) {
        // サービスアカウントでの認証に失敗した場合は、gcloud CLIにフォールバック
        // エラーログは出力しない（正常なフォールバック動作のため）
      }
    }

    // APIキーがない場合は、外部ツールを使用してアクセストークンを取得
    return await this.getOAuthToken();
  }

  private async getOAuthToken(): Promise<Record<string, string>> {
    // gcloud CLIを使用してアクセストークンを取得
    try {
      const process = new Deno.Command("gcloud", {
        args: ["auth", "application-default", "print-access-token"],
        stdout: "piped",
        stderr: "piped",
      });

      const { code, stdout, stderr } = await process.output();

      if (code !== 0) {
        const errorText = new TextDecoder().decode(stderr);
        throw new Error(`gcloud認証エラー: ${errorText}`);
      }

      const accessToken = new TextDecoder().decode(stdout).trim();
      
      return {
        "Authorization": `Bearer ${accessToken}`,
      };
    } catch (error) {
      console.error("gcloud CLIでの認証に失敗しました:", error);
      throw new Error(
        "認証に失敗しました。以下のいずれかの方法で認証してください:\n" +
        "1. 環境変数 GOOGLE_API_KEY にAPIキーを設定\n" +
        "2. gcloud auth application-default login を実行\n" +
        "3. サービスアカウントで直接認証（実装予定）"
      );
    }
  }
}