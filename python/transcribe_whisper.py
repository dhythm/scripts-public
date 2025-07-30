#!/usr/bin/env python3
"""
OpenAI Whisperを使用した音声文字起こしツール

使用方法:
    python transcribe_whisper.py <音声ファイル> [オプション]

例:
    python transcribe_whisper.py input.mp3 --model medium --language ja
    python transcribe_whisper.py input.wav --output json --timestamps
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import whisper
from pydub import AudioSegment
from tqdm import tqdm


class AudioTranscriber:
    """OpenAI Whisperを使用した音声文字起こしクラス"""
    
    SUPPORTED_FORMATS = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma'}
    MODEL_SIZES = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
    
    def __init__(self, model_name: str = 'base'):
        """
        Args:
            model_name: 使用するWhisperモデルのサイズ
        """
        if model_name not in self.MODEL_SIZES:
            raise ValueError(f"無効なモデルサイズ: {model_name}. 有効なオプション: {self.MODEL_SIZES}")
        
        print(f"モデル '{model_name}' をロード中...")
        self.model = whisper.load_model(model_name)
        self.model_name = model_name
        print(f"モデル '{model_name}' のロードが完了しました。")
    
    def validate_audio_file(self, file_path: Path) -> None:
        """音声ファイルの検証"""
        if not file_path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"パスがファイルではありません: {file_path}")
        
        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"サポートされていないファイル形式: {file_path.suffix}\n"
                f"サポートされている形式: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        # ファイルサイズチェック（1GB以上は警告）
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 1024:
            print(f"警告: ファイルサイズが大きいです ({file_size_mb:.1f} MB)。処理に時間がかかる可能性があります。")
    
    def convert_to_wav_if_needed(self, file_path: Path) -> Path:
        """必要に応じてWAV形式に変換"""
        if file_path.suffix.lower() == '.wav':
            return file_path
        
        print(f"音声ファイルをWAV形式に変換中...")
        audio = AudioSegment.from_file(str(file_path))
        
        # 一時ファイルパスを作成
        wav_path = file_path.with_suffix('.wav')
        temp_wav_path = wav_path.parent / f"_temp_{wav_path.name}"
        
        # WAV形式でエクスポート
        audio.export(str(temp_wav_path), format='wav')
        print("変換が完了しました。")
        
        return temp_wav_path
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        verbose: bool = False,
        **kwargs
    ) -> Dict:
        """
        音声ファイルを文字起こし
        
        Args:
            audio_path: 音声ファイルのパス
            language: 言語コード（例: 'ja', 'en'）。Noneの場合は自動検出
            verbose: 詳細な進捗を表示
            **kwargs: その他のWhisperパラメータ
        
        Returns:
            文字起こし結果の辞書
        """
        audio_path = Path(audio_path)
        self.validate_audio_file(audio_path)
        
        # 一時ファイルパスを保持
        temp_file = None
        
        try:
            # 必要に応じてWAV形式に変換
            wav_path = self.convert_to_wav_if_needed(audio_path)
            if wav_path != audio_path:
                temp_file = wav_path
            
            # 文字起こし実行
            print(f"文字起こしを開始します...")
            if language:
                print(f"言語: {language}")
            else:
                print("言語を自動検出します...")
            
            # Whisperのオプション設定
            options = {
                'language': language,
                'verbose': verbose,
                'fp16': False,  # CPU環境でのエラーを避けるため
            }
            options.update(kwargs)
            
            # 文字起こし実行
            result = self.model.transcribe(str(wav_path), **options)
            
            # 検出された言語を表示
            if not language and 'language' in result:
                print(f"検出された言語: {result['language']}")
            
            print("文字起こしが完了しました。")
            
            return result
            
        finally:
            # 一時ファイルの削除
            if temp_file and temp_file.exists():
                temp_file.unlink()
    
    def save_result(
        self,
        result: Dict,
        output_path: Path,
        output_format: str = 'text',
        include_timestamps: bool = False
    ) -> None:
        """
        文字起こし結果を保存
        
        Args:
            result: Whisperの文字起こし結果
            output_path: 出力ファイルパス
            output_format: 出力形式 ('text', 'json', 'srt')
            include_timestamps: タイムスタンプを含めるか
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_format == 'text':
            # テキスト形式で保存
            with output_path.open('w', encoding='utf-8') as f:
                if include_timestamps and 'segments' in result:
                    for segment in result['segments']:
                        start = self._format_timestamp(segment['start'])
                        end = self._format_timestamp(segment['end'])
                        text = segment['text'].strip()
                        f.write(f"[{start} --> {end}] {text}\n")
                else:
                    f.write(result['text'].strip())
        
        elif output_format == 'json':
            # JSON形式で保存
            output_data = {
                'text': result['text'],
                'language': result.get('language', 'unknown'),
                'model': self.model_name,
            }
            
            if include_timestamps and 'segments' in result:
                output_data['segments'] = [
                    {
                        'start': seg['start'],
                        'end': seg['end'],
                        'text': seg['text'].strip(),
                    }
                    for seg in result['segments']
                ]
            
            with output_path.open('w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        elif output_format == 'srt':
            # SRT字幕形式で保存
            with output_path.open('w', encoding='utf-8') as f:
                if 'segments' in result:
                    for i, segment in enumerate(result['segments'], 1):
                        start = self._format_timestamp_srt(segment['start'])
                        end = self._format_timestamp_srt(segment['end'])
                        text = segment['text'].strip()
                        
                        f.write(f"{i}\n")
                        f.write(f"{start} --> {end}\n")
                        f.write(f"{text}\n\n")
        
        print(f"結果を保存しました: {output_path}")
    
    def _format_timestamp(self, seconds: float) -> str:
        """秒をHH:MM:SS形式に変換"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
    
    def _format_timestamp_srt(self, seconds: float) -> str:
        """秒をSRT形式のタイムスタンプに変換"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='OpenAI Whisperを使用した音声文字起こしツール',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使用方法
  %(prog)s input.mp3
  
  # モデルと言語を指定
  %(prog)s input.mp3 --model medium --language ja
  
  # JSON形式でタイムスタンプ付きで出力
  %(prog)s input.wav --output result.json --format json --timestamps
  
  # SRT字幕形式で出力
  %(prog)s input.mp4 --output subtitles.srt --format srt
        """
    )
    
    parser.add_argument(
        'audio_file',
        type=Path,
        help='文字起こしする音声ファイル'
    )
    
    parser.add_argument(
        '--model',
        choices=AudioTranscriber.MODEL_SIZES,
        default='base',
        help='使用するWhisperモデルのサイズ (default: base)'
    )
    
    parser.add_argument(
        '--language',
        type=str,
        help='音声の言語コード (例: ja, en)。指定しない場合は自動検出'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help='出力ファイルパス。指定しない場合は入力ファイル名を基に自動生成'
    )
    
    parser.add_argument(
        '--format',
        choices=['text', 'json', 'srt'],
        default='text',
        help='出力形式 (default: text)'
    )
    
    parser.add_argument(
        '--timestamps',
        action='store_true',
        help='タイムスタンプを含める（text, json形式で有効）'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='詳細な進捗を表示'
    )
    
    parser.add_argument(
        '--task',
        choices=['transcribe', 'translate'],
        default='transcribe',
        help='実行するタスク。translateを指定すると英語に翻訳 (default: transcribe)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0,
        help='サンプリング温度 (0-1)。高いほど多様な出力 (default: 0)'
    )
    
    args = parser.parse_args()
    
    try:
        # 出力ファイルパスを決定
        if args.output:
            output_path = args.output
        else:
            # 拡張子を適切に設定
            ext_map = {'text': '.txt', 'json': '.json', 'srt': '.srt'}
            ext = ext_map[args.format]
            output_path = args.audio_file.with_suffix(ext)
        
        # 文字起こし実行
        transcriber = AudioTranscriber(model_name=args.model)
        
        # Whisperオプションの設定
        whisper_options = {
            'task': args.task,
            'temperature': args.temperature,
        }
        
        # 文字起こし実行
        result = transcriber.transcribe(
            args.audio_file,
            language=args.language,
            verbose=args.verbose,
            **whisper_options
        )
        
        # 結果を保存
        transcriber.save_result(
            result,
            output_path,
            output_format=args.format,
            include_timestamps=args.timestamps or args.format == 'srt'
        )
        
        # 簡単な統計情報を表示
        print(f"\n--- 統計情報 ---")
        print(f"文字数: {len(result['text'])} 文字")
        if 'segments' in result:
            print(f"セグメント数: {len(result['segments'])}")
            duration = result['segments'][-1]['end'] if result['segments'] else 0
            print(f"音声の長さ: {transcriber._format_timestamp(duration)}")
        
    except KeyboardInterrupt:
        print("\n処理が中断されました。")
        sys.exit(1)
    except Exception as e:
        print(f"\nエラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()