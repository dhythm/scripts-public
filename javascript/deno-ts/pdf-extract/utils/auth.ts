export interface GoogleAuthConfig {
  credentials?: string;
  projectId?: string;
  apiKey?: string;
}

export interface ServiceAccountKey {
  type: string;
  project_id: string;
  private_key_id: string;
  private_key: string;
  client_email: string;
  client_id: string;
  auth_uri: string;
  token_uri: string;
  auth_provider_x509_cert_url: string;
  client_x509_cert_url: string;
}

export class GoogleAuthHelper {
  private credentials?: ServiceAccountKey;
  private projectId?: string;
  private accessToken?: string;
  private tokenExpiry?: Date;
  private verbose: boolean = false;

  constructor(verbose: boolean = false) {
    this.verbose = verbose;
  }

  async authenticate(): Promise<void> {
    const credentialsPath = Deno.env.get("GOOGLE_APPLICATION_CREDENTIALS");
    
    if (!credentialsPath) {
      throw new Error(
        "環境変数 GOOGLE_APPLICATION_CREDENTIALS が設定されていません。\n" +
        "Google Cloud サービスアカウントキーのパスを設定してください。"
      );
    }

    try {
      const keyFile = await Deno.readTextFile(credentialsPath);
      this.credentials = JSON.parse(keyFile) as ServiceAccountKey;
      
      // サービスアカウントキーからproject_idを取得
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
    } catch (error) {
      if (error instanceof Deno.errors.NotFound) {
        throw new Error(
          `認証情報ファイルが見つかりません: ${credentialsPath}\n` +
          "ファイルパスが正しいことを確認してください。"
        );
      }
      if (error instanceof SyntaxError) {
        throw new Error(
          "認証情報ファイルの形式が正しくありません。\n" +
          "有効なJSONファイルであることを確認してください。"
        );
      }
      throw error;
    }
  }

  async getAccessToken(): Promise<string> {
    if (this.accessToken && this.tokenExpiry && this.tokenExpiry > new Date()) {
      return this.accessToken;
    }

    if (!this.credentials) {
      throw new Error("認証情報が設定されていません。先にauthenticate()を実行してください。");
    }

    const jwt = await this.createJWT();
    
    const tokenResponse = await this.exchangeJWTForToken(jwt);
    
    // Google OAuth2 APIのレスポンスからaccess_tokenを取得
    if (tokenResponse.access_token) {
      this.accessToken = tokenResponse.access_token;
      this.tokenExpiry = new Date(Date.now() + (tokenResponse.expires_in - 60) * 1000);
    } else if (tokenResponse.id_token) {
      // id_tokenが返された場合はエラー（verboseモードでのみ表示）
      if (this.verbose) {
        console.debug("エラー: id_tokenが返されました。access_tokenが必要です。");
        console.debug("トークンレスポンス:", tokenResponse);
      }
      throw new Error("認証エラー: access_tokenではなくid_tokenが返されました");
    } else {
      if (this.verbose) {
        console.log("トークン取得失敗:", tokenResponse);
      }
      throw new Error("認証エラー: access_tokenが取得できませんでした");
    }
    
    return this.accessToken!;
  }

  getProjectId(): string {
    if (!this.projectId && this.credentials?.project_id) {
      return this.credentials.project_id;
    }
    if (!this.projectId) {
      throw new Error("プロジェクトIDが設定されていません。");
    }
    return this.projectId;
  }

  private async createJWT(): Promise<string> {
    if (!this.credentials) {
      throw new Error("認証情報が設定されていません。");
    }

    const header = {
      alg: "RS256",
      typ: "JWT",
    };

    const now = Math.floor(Date.now() / 1000);
    const payload = {
      iss: this.credentials.client_email,
      scope: [
        "https://www.googleapis.com/auth/cloud-vision",
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/documentai",
      ].join(" "),
      aud: "https://oauth2.googleapis.com/token",
      exp: now + 3600,
      iat: now,
    };

    const encodedHeader = this.base64urlEncode(JSON.stringify(header));
    const encodedPayload = this.base64urlEncode(JSON.stringify(payload));
    const signatureInput = `${encodedHeader}.${encodedPayload}`;

    const signature = await this.signRS256(signatureInput, this.credentials.private_key);
    
    return `${signatureInput}.${signature}`;
  }

  private async exchangeJWTForToken(jwt: string): Promise<any> {
    if (!this.credentials) {
      throw new Error("認証情報が設定されていません。");
    }

    
    const response = await fetch(this.credentials.token_uri, {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: new URLSearchParams({
        grant_type: "urn:ietf:params:oauth:grant-type:jwt-bearer",
        assertion: jwt,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`トークン取得エラー: ${error}`);
    }

    const result = await response.json();
    return result;
  }

  private base64urlEncode(data: string): string {
    const encoder = new TextEncoder();
    const encoded = btoa(String.fromCharCode(...encoder.encode(data)));
    return encoded.replace(/\+/g, "-").replace(/\//g, "_").replace(/=/g, "");
  }

  private async signRS256(data: string, privateKey: string): Promise<string> {
    const keyData = privateKey
      .replace(/-----BEGIN PRIVATE KEY-----/g, "")
      .replace(/-----END PRIVATE KEY-----/g, "")
      .replace(/\s/g, "");

    const binaryKey = Uint8Array.from(atob(keyData), c => c.charCodeAt(0));

    const cryptoKey = await crypto.subtle.importKey(
      "pkcs8",
      binaryKey,
      {
        name: "RSASSA-PKCS1-v1_5",
        hash: "SHA-256",
      },
      false,
      ["sign"]
    );

    const encoder = new TextEncoder();
    const signature = await crypto.subtle.sign(
      "RSASSA-PKCS1-v1_5",
      cryptoKey,
      encoder.encode(data)
    );

    const base64Signature = btoa(String.fromCharCode(...new Uint8Array(signature)));
    return base64Signature.replace(/\+/g, "-").replace(/\//g, "_").replace(/=/g, "");
  }
}