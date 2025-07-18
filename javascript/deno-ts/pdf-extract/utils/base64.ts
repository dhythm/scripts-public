export function encodeBase64Chunked(data: Uint8Array): string {
  const CHUNK_SIZE = 1024 * 1024; // 1MB chunks
  const chunks: string[] = [];
  
  for (let i = 0; i < data.length; i += CHUNK_SIZE) {
    const chunk = data.slice(i, i + CHUNK_SIZE);
    const binary = Array.from(chunk, (byte) => String.fromCharCode(byte)).join('');
    chunks.push(btoa(binary));
  }
  
  return chunks.join('');
}

export function encodeBase64Stream(data: Uint8Array): string {
  // より効率的なBase64エンコーディング
  const buffer = new ArrayBuffer(data.length);
  const view = new Uint8Array(buffer);
  view.set(data);
  
  // バイナリ文字列を作成する際にチャンクで処理
  const CHUNK_SIZE = 8192;
  let binaryString = '';
  
  for (let i = 0; i < view.length; i += CHUNK_SIZE) {
    const chunk = view.subarray(i, Math.min(i + CHUNK_SIZE, view.length));
    binaryString += String.fromCharCode.apply(null, Array.from(chunk));
  }
  
  return btoa(binaryString);
}