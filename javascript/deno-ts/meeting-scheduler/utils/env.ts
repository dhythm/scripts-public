import { load } from "https://deno.land/std@0.220.0/dotenv/mod.ts";
import * as path from "https://deno.land/std@0.220.0/path/mod.ts";

/**
 * .envファイルから環境変数を読み込む
 */
export async function loadEnv(): Promise<void> {
  // meeting-schedulerディレクトリの.envファイルを探す
  const currentDir = Deno.cwd();
  let envPath = "./meeting-scheduler/.env";
  
  // meeting-schedulerディレクトリから実行された場合
  if (currentDir.endsWith("meeting-scheduler")) {
    envPath = "./.env";
  }
  // 絶対パスで指定する別の方法
  else if (!await fileExists(envPath)) {
    // javascript/deno-tsから実行されていることを想定
    envPath = path.join(currentDir, "meeting-scheduler", ".env");
  }
  
  try {
    const envVars = await load({
      envPath,
      export: true,
      allowEmptyValues: true,
    });
    
    // デバッグ用
    if (Deno.env.get("DEBUG") === "true") {
      console.log(`✅ .envファイルを読み込みました: ${envPath}`);
      const keys = Object.keys(envVars);
      console.log(`  読み込んだ環境変数: ${keys.length}個`);
      console.log(`  変数名: ${keys.join(", ")}`);
    }
  } catch (error) {
    // .envファイルが存在しない場合は警告のみ
    if (error instanceof Deno.errors.NotFound) {
      console.warn(`⚠️ .envファイルが見つかりません: ${envPath}`);
      console.warn("環境変数が設定されていることを確認してください。");
      console.warn("ヒント: meeting-scheduler/.env.exampleをコピーして.envを作成してください。");
    } else {
      throw error;
    }
  }
}

async function fileExists(path: string): Promise<boolean> {
  try {
    await Deno.stat(path);
    return true;
  } catch {
    return false;
  }
}

/**
 * 必須の環境変数が設定されているか確認
 */
export function validateEnv(requiredVars: string[]): void {
  const missingVars: string[] = [];
  
  for (const varName of requiredVars) {
    if (!Deno.env.get(varName)) {
      missingVars.push(varName);
    }
  }
  
  if (missingVars.length > 0) {
    throw new Error(
      `必須の環境変数が設定されていません:\n` +
      missingVars.map(v => `  - ${v}`).join("\n") +
      `\n\n.env.exampleを参考に.envファイルを作成してください。`
    );
  }
}

/**
 * 環境変数を取得（デフォルト値付き）
 */
export function getEnvVar(name: string, defaultValue?: string): string {
  const value = Deno.env.get(name);
  if (value === undefined && defaultValue === undefined) {
    throw new Error(`環境変数 ${name} が設定されていません。`);
  }
  return value || defaultValue!;
}