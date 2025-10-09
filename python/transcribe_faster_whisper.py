#!/usr/bin/env python3
"""
Faster-Whisperを使用した音声文字起こしツール

使用方法:
    python transcribe_faster_whisper.py <音声ファイル> [オプション]

例:
    python transcribe_faster_whisper.py input.mp3 --model large-v2 --language ja
    python transcribe_faster_whisper.py input.wav --device cuda --compute_type float16
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from faster_whisper import WhisperModel
from pydub import AudioSegment
from tqdm import tqdm

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False


class FasterWhisperTranscriber:
    """Faster-Whisperを使用した音声文字起こしクラス"""
    
    SUPPORTED_FORMATS = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma'}
    MODEL_SIZES = [
        'tiny', 'tiny.en', 'base', 'base.en', 
        'small', 'small.en', 'medium', 'medium.en',
        'large-v1', 'large-v2', 'large-v3', 'large'
    ]
    COMPUTE_TYPES = ['default', 'float16', 'int8_float16', 'int8']
    
    def __init__(
        self,
        model_name: str = 'base',
        device: str = 'auto',
        compute_type: str = 'default',
        cpu_threads: int = 0,
        num_workers: int = 1
    ):
        """
        Args:
            model_name: 使用するWhisperモデルのサイズ
            device: 使用するデバイス ('cuda', 'cpu', 'auto')
            compute_type: 計算精度
            cpu_threads: CPU使用時のスレッド数（0で自動）
            num_workers: 並列ワーカー数
        """
        if model_name not in self.MODEL_SIZES:
            raise ValueError(f"無効なモデルサイズ: {model_name}. 有効なオプション: {self.MODEL_SIZES}")
        
        # デバイスの決定
        if device == 'auto':
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print(f"自動検出されたデバイス: {device}")
            except ImportError:
                device = 'cpu'
                print("PyTorchがインストールされていません。CPUを使用します。")
        
        print(f"モデル '{model_name}' をロード中...")
        print(f"デバイス: {device}, 計算精度: {compute_type}")
        
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers
        )
        
        self.model_name = model_name
        self.device = device
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
    
    def apply_noise_reduction(
        self,
        audio: AudioSegment,
        reduce_amount: float = 0.8,
        stationary: bool = False
    ) -> AudioSegment:
        """
        ノイズ除去を適用

        Args:
            audio: AudioSegmentオブジェクト
            reduce_amount: ノイズ除去の強度（0.0-1.0）。1.0が最大
            stationary: 定常ノイズ（ファンの音など）の場合はTrue

        Returns:
            ノイズ除去後のAudioSegment
        """
        if not NOISEREDUCE_AVAILABLE:
            print("警告: noisereduceがインストールされていません。ノイズ除去をスキップします。")
            return audio

        print(f"ノイズ除去を実行中... (強度: {reduce_amount:.1f}, 定常ノイズ: {stationary})")

        # AudioSegmentをnumpy配列に変換
        samples = np.array(audio.get_array_of_samples())

        # ステレオの場合はモノラルに変換
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
            samples = samples.mean(axis=1)

        # float32に正規化
        samples = samples.astype(np.float32)
        if audio.sample_width == 2:  # 16-bit
            samples = samples / 32768.0
        elif audio.sample_width == 1:  # 8-bit
            samples = samples / 128.0

        # ノイズ除去を適用
        # 最初の0.5秒をノイズサンプルとして使用
        sample_rate = audio.frame_rate
        noise_duration = min(0.5, len(samples) / sample_rate / 2)  # 最大0.5秒
        noise_len = int(noise_duration * sample_rate)

        if noise_len > 0:
            y_noise = samples[:noise_len]
            reduced = nr.reduce_noise(
                y=samples,
                sr=sample_rate,
                y_noise=y_noise,
                stationary=stationary,
                prop_decrease=reduce_amount
            )
        else:
            # ノイズサンプルなしで実行
            reduced = nr.reduce_noise(
                y=samples,
                sr=sample_rate,
                stationary=stationary,
                prop_decrease=reduce_amount
            )

        # 16-bit intに戻す
        reduced = (reduced * 32768.0).astype(np.int16)

        # AudioSegmentに戻す
        denoised_audio = AudioSegment(
            reduced.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )

        print("ノイズ除去が完了しました。")
        return denoised_audio

    def convert_to_wav_if_needed(
        self,
        file_path: Path,
        normalize: bool = False,
        gain_db: float = 0,
        denoise: bool = False,
        noise_reduce_amount: float = 0.8
    ) -> Path:
        """
        必要に応じてWAV形式に変換し、音声を前処理

        Args:
            file_path: 音声ファイルのパス
            normalize: ピークノーマライズを適用するか
            gain_db: ゲイン調整(dB)。正の値で音量アップ、負の値で音量ダウン
            denoise: ノイズ除去を適用するか
            noise_reduce_amount: ノイズ除去の強度（0.0-1.0）

        Returns:
            変換後のWAVファイルのパス
        """
        # WAVファイルでも前処理が必要な場合は読み込む
        needs_processing = normalize or gain_db != 0 or denoise
        if file_path.suffix.lower() == '.wav' and not needs_processing:
            return file_path

        print(f"音声ファイルをWAV形式に変換中...")
        audio = AudioSegment.from_file(str(file_path))

        # ノイズ除去（他の処理の前に実行）
        if denoise:
            audio = self.apply_noise_reduction(
                audio,
                reduce_amount=noise_reduce_amount,
                stationary=False  # 非定常ノイズ対応
            )

        # ゲイン調整
        if gain_db != 0:
            print(f"ゲイン調整: {gain_db:+.1f} dB")
            audio = audio + gain_db

        # ノーマライズ（ピークを-3.0 dBFSに調整）
        if normalize:
            target_dBFS = -3.0
            change_in_dBFS = target_dBFS - audio.dBFS
            print(f"ノーマライズ: {change_in_dBFS:+.1f} dB (ピーク: {audio.dBFS:.1f} → {target_dBFS:.1f} dBFS)")
            audio = audio.apply_gain(change_in_dBFS)

        # 16kHzにリサンプリング（Whisperの推奨）
        audio = audio.set_frame_rate(16000)

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
        task: str = 'transcribe',
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1.0,
        length_penalty: float = 1.0,
        temperature: Union[float, List[float]] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        compression_ratio_threshold: float = 2.4,
        log_prob_threshold: float = -1.0,
        no_speech_threshold: float = 0.6,
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = False,
        vad_filter: bool = True,
        vad_parameters: Optional[Dict] = None,
        normalize: bool = False,
        gain_db: float = 0,
        denoise: bool = False,
        noise_reduce_amount: float = 0.8
    ) -> Dict:
        """
        音声ファイルを文字起こし

        Args:
            audio_path: 音声ファイルのパス
            language: 言語コード（例: 'ja', 'en'）。Noneの場合は自動検出
            task: 'transcribe' または 'translate'
            beam_size: ビームサーチのビーム幅
            best_of: 候補数
            patience: ビームサーチのpatience
            length_penalty: 長さペナルティ
            temperature: サンプリング温度
            compression_ratio_threshold: 圧縮率の閾値
            log_prob_threshold: ログ確率の閾値
            no_speech_threshold: 無音検出の閾値
            initial_prompt: 初期プロンプト
            word_timestamps: 単語レベルのタイムスタンプを生成
            vad_filter: Voice Activity Detectionフィルタを使用
            vad_parameters: VADパラメータ
            normalize: 音声をピークノーマライズ
            gain_db: ゲイン調整(dB)
            denoise: ノイズ除去を適用
            noise_reduce_amount: ノイズ除去の強度（0.0-1.0）

        Returns:
            文字起こし結果の辞書
        """
        audio_path = Path(audio_path)
        self.validate_audio_file(audio_path)

        # デフォルトVADパラメータ（小さい音声に最適化）
        if vad_parameters is None and vad_filter:
            vad_parameters = {
                'threshold': 0.30,              # 小さい音声を検出（デフォルト0.5→0.3）
                'min_speech_duration_ms': 100,  # 短い発話も保持（デフォルト0→100）
                'min_silence_duration_ms': 1000,# 短い無音で分割しない（デフォルト2000→1000）
                'speech_pad_ms': 400            # 音声の前後を保護（デフォルト400）
            }
            print(f"VADパラメータ（小さい音声用）: threshold={vad_parameters['threshold']}")

        # 一時ファイルパスを保持
        temp_file = None

        try:
            # 必要に応じてWAV形式に変換（前処理を含む）
            wav_path = self.convert_to_wav_if_needed(
                audio_path,
                normalize=normalize,
                gain_db=gain_db,
                denoise=denoise,
                noise_reduce_amount=noise_reduce_amount
            )
            if wav_path != audio_path:
                temp_file = wav_path
            
            # 文字起こし実行
            print(f"文字起こしを開始します...")
            if language:
                print(f"言語: {language}")
            else:
                print("言語を自動検出します...")
            
            # 文字起こし実行
            segments, info = self.model.transcribe(
                str(wav_path),
                language=language,
                task=task,
                beam_size=beam_size,
                best_of=best_of,
                patience=patience,
                length_penalty=length_penalty,
                temperature=temperature,
                compression_ratio_threshold=compression_ratio_threshold,
                log_prob_threshold=log_prob_threshold,
                no_speech_threshold=no_speech_threshold,
                initial_prompt=initial_prompt,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter,
                vad_parameters=vad_parameters
            )
            
            # セグメントをリストに変換（プログレスバー付き）
            segment_list = []
            full_text = []
            
            print("セグメントを処理中...")
            for segment in tqdm(segments, desc="セグメント"):
                seg_dict = {
                    'id': segment.id,
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text,
                    'no_speech_prob': segment.no_speech_prob,
                    'avg_logprob': segment.avg_logprob,
                    'compression_ratio': segment.compression_ratio,
                }
                
                # 単語レベルのタイムスタンプを追加
                if word_timestamps and segment.words:
                    seg_dict['words'] = [
                        {
                            'word': word.word,
                            'start': word.start,
                            'end': word.end,
                            'probability': word.probability
                        }
                        for word in segment.words
                    ]
                
                segment_list.append(seg_dict)
                full_text.append(segment.text)
            
            # 検出された言語を表示
            if info.language:
                print(f"検出された言語: {info.language}")
            
            print("文字起こしが完了しました。")
            
            # 結果を辞書形式で返す
            result = {
                'text': ''.join(full_text),
                'segments': segment_list,
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration,
                'duration_after_vad': info.duration_after_vad,
                'all_language_probs': info.all_language_probs
            }
            
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
        include_timestamps: bool = False,
        include_confidence: bool = False
    ) -> None:
        """
        文字起こし結果を保存
        
        Args:
            result: 文字起こし結果
            output_path: 出力ファイルパス
            output_format: 出力形式 ('text', 'json', 'srt', 'vtt')
            include_timestamps: タイムスタンプを含めるか
            include_confidence: 信頼度情報を含めるか
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
                        
                        if include_confidence:
                            conf = f" [信頼度: {1 - segment['no_speech_prob']:.2f}]"
                            f.write(f"[{start} --> {end}]{conf} {text}\n")
                        else:
                            f.write(f"[{start} --> {end}] {text}\n")
                else:
                    f.write(result['text'].strip())
        
        elif output_format == 'json':
            # JSON形式で保存
            output_data = {
                'text': result['text'],
                'language': result.get('language', 'unknown'),
                'language_probability': result.get('language_probability', 0),
                'duration': result.get('duration', 0),
                'model': self.model_name,
                'device': self.device,
            }
            
            if include_timestamps and 'segments' in result:
                segments_data = []
                for seg in result['segments']:
                    seg_data = {
                        'id': seg['id'],
                        'start': seg['start'],
                        'end': seg['end'],
                        'text': seg['text'].strip(),
                    }
                    
                    if include_confidence:
                        seg_data.update({
                            'no_speech_prob': seg['no_speech_prob'],
                            'avg_logprob': seg['avg_logprob'],
                            'compression_ratio': seg['compression_ratio'],
                        })
                    
                    if 'words' in seg:
                        seg_data['words'] = seg['words']
                    
                    segments_data.append(seg_data)
                
                output_data['segments'] = segments_data
            
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
        
        elif output_format == 'vtt':
            # WebVTT字幕形式で保存
            with output_path.open('w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
                
                if 'segments' in result:
                    for segment in result['segments']:
                        start = self._format_timestamp_vtt(segment['start'])
                        end = self._format_timestamp_vtt(segment['end'])
                        text = segment['text'].strip()
                        
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
    
    def _format_timestamp_vtt(self, seconds: float) -> str:
        """秒をWebVTT形式のタイムスタンプに変換"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='Faster-Whisperを使用した音声文字起こしツール',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使用方法
  %(prog)s input.mp3
  
  # モデルと言語を指定
  %(prog)s input.mp3 --model large-v2 --language ja
  
  # GPUを使用して高速処理
  %(prog)s input.wav --device cuda --compute_type float16
  
  # 単語レベルのタイムスタンプ付きJSON出力
  %(prog)s input.mp4 --format json --word_timestamps --output result.json
  
  # WebVTT字幕形式で出力
  %(prog)s input.mp3 --format vtt --output subtitles.vtt
        """
    )
    
    parser.add_argument(
        'audio_file',
        type=Path,
        help='文字起こしする音声ファイル'
    )
    
    parser.add_argument(
        '--model',
        choices=FasterWhisperTranscriber.MODEL_SIZES,
        default='base',
        help='使用するWhisperモデルのサイズ (default: base)'
    )
    
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='使用するデバイス (default: auto)'
    )
    
    parser.add_argument(
        '--compute_type',
        choices=FasterWhisperTranscriber.COMPUTE_TYPES,
        default='default',
        help='計算精度 (default: default)'
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
        choices=['text', 'json', 'srt', 'vtt'],
        default='text',
        help='出力形式 (default: text)'
    )
    
    parser.add_argument(
        '--timestamps',
        action='store_true',
        help='タイムスタンプを含める（text, json形式で有効）'
    )
    
    parser.add_argument(
        '--word_timestamps',
        action='store_true',
        help='単語レベルのタイムスタンプを生成（処理時間が増加）'
    )
    
    parser.add_argument(
        '--confidence',
        action='store_true',
        help='信頼度情報を含める（text, json形式で有効）'
    )
    
    parser.add_argument(
        '--task',
        choices=['transcribe', 'translate'],
        default='transcribe',
        help='実行するタスク。translateを指定すると英語に翻訳 (default: transcribe)'
    )
    
    parser.add_argument(
        '--beam_size',
        type=int,
        default=5,
        help='ビームサーチのビーム幅 (default: 5)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        nargs='+',
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help='サンプリング温度 (default: 0.0 0.2 0.4 0.6 0.8 1.0)'
    )
    
    parser.add_argument(
        '--vad_filter',
        action='store_true',
        default=True,
        help='Voice Activity Detectionフィルタを使用 (default: True)'
    )

    parser.add_argument(
        '--no_vad_filter',
        dest='vad_filter',
        action='store_false',
        help='Voice Activity Detectionフィルタを無効化'
    )

    parser.add_argument(
        '--vad_threshold',
        type=float,
        help='VAD閾値（0-1）。低いほど敏感（小さい音声を検出）。指定しない場合は0.30'
    )

    parser.add_argument(
        '--vad_min_speech_duration',
        type=int,
        help='最小音声長(ms)。これより短い音声は無視'
    )

    parser.add_argument(
        '--vad_max_speech_duration',
        type=float,
        help='最大音声長(s)。これより長い場合は分割'
    )

    parser.add_argument(
        '--vad_min_silence_duration',
        type=int,
        help='最小無音長(ms)。これより短い無音では分割しない'
    )

    parser.add_argument(
        '--vad_speech_pad',
        type=int,
        help='音声の前後パディング(ms)。音声の切れを防ぐ'
    )

    parser.add_argument(
        '--normalize',
        action='store_true',
        help='音声をピークノーマライズ（-3.0 dBFSに調整）'
    )

    parser.add_argument(
        '--gain',
        type=float,
        default=0,
        help='ゲイン調整(dB)。正の値で音量アップ、負の値で音量ダウン (default: 0)'
    )

    parser.add_argument(
        '--denoise',
        action='store_true',
        help='ノイズ除去を適用（ノイズが多い環境で推奨）'
    )

    parser.add_argument(
        '--noise_reduce_amount',
        type=float,
        default=0.8,
        help='ノイズ除去の強度（0.0-1.0）。1.0が最大 (default: 0.8)'
    )

    parser.add_argument(
        '--no_speech_threshold',
        type=float,
        default=0.6,
        help='Whisper側の無音判定閾値。1.0に近づけると無音除外を緩和 (default: 0.6)'
    )

    parser.add_argument(
        '--initial_prompt',
        type=str,
        help='文字起こしの初期プロンプト（文脈を与える）'
    )

    parser.add_argument(
        '--cpu_threads',
        type=int,
        default=0,
        help='CPU使用時のスレッド数（0で自動）(default: 0)'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='並列ワーカー数 (default: 1)'
    )
    
    args = parser.parse_args()
    
    try:
        # 出力ファイルパスを決定
        if args.output:
            output_path = args.output
        else:
            # 拡張子を適切に設定
            ext_map = {'text': '.txt', 'json': '.json', 'srt': '.srt', 'vtt': '.vtt'}
            ext = ext_map[args.format]
            output_path = args.audio_file.with_suffix(ext)
        
        # 文字起こし実行
        transcriber = FasterWhisperTranscriber(
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            cpu_threads=args.cpu_threads,
            num_workers=args.num_workers
        )
        
        # 温度パラメータの処理
        temperature = args.temperature[0] if len(args.temperature) == 1 else args.temperature

        # VADパラメータの構築
        vad_parameters = None
        if args.vad_filter:
            vad_parameters = {}
            if args.vad_threshold is not None:
                vad_parameters['threshold'] = args.vad_threshold
            if args.vad_min_speech_duration is not None:
                vad_parameters['min_speech_duration_ms'] = args.vad_min_speech_duration
            if args.vad_max_speech_duration is not None:
                vad_parameters['max_speech_duration_s'] = args.vad_max_speech_duration
            if args.vad_min_silence_duration is not None:
                vad_parameters['min_silence_duration_ms'] = args.vad_min_silence_duration
            if args.vad_speech_pad is not None:
                vad_parameters['speech_pad_ms'] = args.vad_speech_pad
            # パラメータが何も指定されていない場合はNoneに戻す（デフォルト値を使用）
            if not vad_parameters:
                vad_parameters = None

        # 文字起こし実行
        result = transcriber.transcribe(
            args.audio_file,
            language=args.language,
            task=args.task,
            beam_size=args.beam_size,
            temperature=temperature,
            initial_prompt=args.initial_prompt,
            word_timestamps=args.word_timestamps,
            vad_filter=args.vad_filter,
            vad_parameters=vad_parameters,
            normalize=args.normalize,
            gain_db=args.gain,
            denoise=args.denoise,
            noise_reduce_amount=args.noise_reduce_amount,
            no_speech_threshold=args.no_speech_threshold
        )
        
        # 結果を保存
        transcriber.save_result(
            result,
            output_path,
            output_format=args.format,
            include_timestamps=args.timestamps or args.format in ['srt', 'vtt'],
            include_confidence=args.confidence
        )
        
        # 簡単な統計情報を表示
        print(f"\n--- 統計情報 ---")
        print(f"文字数: {len(result['text'])} 文字")
        print(f"セグメント数: {len(result['segments'])}")
        
        if result.get('duration'):
            print(f"音声の長さ: {transcriber._format_timestamp(result['duration'])}")
            if result.get('duration_after_vad'):
                vad_reduction = (1 - result['duration_after_vad'] / result['duration']) * 100
                print(f"VAD後の長さ: {transcriber._format_timestamp(result['duration_after_vad'])} "
                      f"({vad_reduction:.1f}%削減)")
        
        if result.get('language_probability'):
            print(f"言語検出の信頼度: {result['language_probability']:.2%}")
        
    except KeyboardInterrupt:
        print("\n処理が中断されました。")
        sys.exit(1)
    except Exception as e:
        print(f"\nエラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
