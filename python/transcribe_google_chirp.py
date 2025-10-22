#!/usr/bin/env python3
"""Google Cloud Speech-to-Text v2 (Chirp 3) transcription CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import Conflict, Forbidden, NotFound, PermissionDenied
from google.cloud import speech_v2
from google.cloud.speech_v2.types import cloud_speech

try:
    from pydub import AudioSegment
except ImportError:  # pragma: no cover
    AudioSegment = None  # type: ignore

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    import noisereduce as nr
except ImportError:  # pragma: no cover
    nr = None  # type: ignore


@dataclass
class Segment:
    transcript: str
    confidence: Optional[float]
    start_ms: Optional[int]
    end_ms: Optional[int]


@dataclass
class Word:
    word: str
    start_ms: Optional[int]
    end_ms: Optional[int]
    speaker_label: Optional[str]


@dataclass
class TranscriptionResult:
    source: str
    transcript: str
    language_code: Optional[str]
    model: str
    mode: str
    segments: List[Segment] = field(default_factory=list)
    words: List[Word] = field(default_factory=list)


class TranscriptionError(RuntimeError):
    """Wraps recoverable transcription failures."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Google Cloud Speech-to-Text v2 (Chirp 3) 日本語対応文字起こしツール"
    )
    parser.add_argument("audio", nargs="+", help="入力音声ファイルまたは gs:// URI (複数可)")
    parser.add_argument("--project", help="GCP プロジェクト ID (未指定時は GOOGLE_CLOUD_PROJECT を参照)")
    parser.add_argument(
        "--location",
        default="asia-northeast1",
        help="Chirp 3 対応リージョン (デフォルト: asia-northeast1=東京)",
    )
    parser.add_argument("--recognizer", default="_", help="使用する Recognizer ID (デフォルトは暗黙の _)")
    parser.add_argument("--model", default="chirp_3", help="使用するモデル ID")
    parser.add_argument(
        "--mode",
        choices=("auto", "sync", "streaming", "batch"),
        default="auto",
        help="処理モード (auto はファイルサイズで同期/バッチを自動選択)",
    )
    parser.add_argument(
        "--sync-max-bytes",
        type=int,
        default=10 * 1024 * 1024,
        help="auto モードで同期認識を使う最大バイト数",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=15_000,
        help="ストリーミング時のチャンクサイズ (最大 15 KB を推奨)",
    )
    parser.add_argument(
        "--language-codes",
        nargs="+",
        default=["ja-JP"],
        help="期待する言語コード (複数指定可)",
    )
    parser.add_argument(
        "--auto-language",
        action="store_true",
        help="言語自動検出を有効化 (language_codes を上書きして ['auto'] を使用)",
    )
    parser.add_argument("--max-alternatives", type=int, default=1, help="候補出力数")
    parser.add_argument("--profanity-filter", action="store_true", help="NG ワードを伏字化")
    parser.add_argument(
        "--enable-word-time-offsets",
        action="store_true",
        help="単語レベルのタイムスタンプ出力を要求",
    )
    parser.add_argument(
        "--disable-auto-punctuation",
        action="store_true",
        help="自動句読点を無効化",
    )
    parser.add_argument(
        "--enable-spoken-punctuation",
        action="store_true",
        help="話し言葉の句読点を記号に変換",
    )
    parser.add_argument(
        "--enable-spoken-emojis", action="store_true", help="話し言葉の絵文字を記号化"
    )
    parser.add_argument(
        "--denoise-audio",
        action="store_true",
        help="API 内蔵のデノイザーを有効化",
    )
    parser.add_argument(
        "--snr-threshold",
        type=float,
        help="SNR フィルタ閾値 (denoise と併用可)",
    )
    parser.add_argument(
        "--phrase",
        action="append",
        default=[],
        help="適応 (Speech Adaptation) 用のフレーズ。複数指定可",
    )
    parser.add_argument(
        "--phrase-boost",
        type=float,
        default=10.0,
        help="適応フレーズのブースト値 (0-20 推奨)",
    )
    parser.add_argument(
        "--interim-results",
        action="store_true",
        help="ストリーミング時に途中結果を標準エラーへ表示",
    )
    parser.add_argument(
        "--upload-bucket",
        help="Batch モード時にローカル音声をアップロードする Cloud Storage バケット名",
    )
    parser.add_argument(
        "--upload-prefix",
        default="transcribe",
        help="Cloud Storage アップロード時のオブジェクト接頭辞",
    )
    parser.add_argument(
        "--batch-timeout",
        type=int,
        default=6 * 3600,
        help="Batch Recognize の待機秒数",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="単一ファイル処理時の出力パス。複数入力の場合は --output-dir を使用",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="複数入力時の個別出力ディレクトリ (存在しなければ作成)",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="出力形式 (text は純テキスト, json は詳細情報含む)",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="実際に送信する RecognitionConfig を標準エラーへ表示",
    )
    parser.add_argument(
        "--preprocess-normalize",
        action="store_true",
        help="ローカル音声にピークノーマライズを適用して音量差を均す",
    )
    parser.add_argument(
        "--preprocess-target-dbfs",
        type=float,
        default=-3.0,
        help="ノーマライズ時の目標ピークレベル (dBFS)",
    )
    parser.add_argument(
        "--preprocess-gain-db",
        type=float,
        default=0.0,
        help="ローカル音声に追加で与えるゲイン量 (dB)",
    )
    parser.add_argument(
        "--preprocess-denoise",
        action="store_true",
        help="ローカル音声にスペクトル減算法ベースのノイズリダクションを適用",
    )
    parser.add_argument(
        "--preprocess-noise-reduce-amount",
        type=float,
        default=0.8,
        help="ノイズリダクションの強度 (0.0-1.0, 大きいほど強く削減)",
    )
    parser.add_argument(
        "--preprocess-stationary-noise",
        action="store_true",
        help="定常ノイズを想定してノイズ推定を安定化 (noisereduce の stationary=True)",
    )
    return parser.parse_args()


SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".wma"}


def is_gcs_uri(value: str) -> bool:
    return value.startswith("gs://")


def sanitize_bucket_component(value: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "-" for ch in value]
    sanitized = "".join(chars).strip("-")
    sanitized = "-".join(filter(None, sanitized.split("-")))
    return sanitized or "default"


def derive_default_bucket_name(project: str, location: str) -> str:
    project_part = sanitize_bucket_component(project)
    location_part = sanitize_bucket_component(location or "us")
    digest = hashlib.sha1(f"{project}:{location}".encode("utf-8")).hexdigest()[:8]
    base_prefix = f"chirp-transcribe-{location_part}" if location_part else "chirp-transcribe"
    max_project_len = max(1, 63 - len(base_prefix) - len(digest) - 2)
    project_part = project_part[:max_project_len].strip("-")
    if project_part:
        name = f"{base_prefix}-{project_part}-{digest}"
    else:
        name = f"{base_prefix}-{digest}"
    return name.strip("-")[:63]


def ensure_bucket_exists(storage_client, bucket_name: str, location: str) -> bool:
    try:
        storage_client.get_bucket(bucket_name)
        return False
    except Forbidden as exc:
        raise TranscriptionError(
            "Cloud Storage バケットへアクセスできません。\n"
            f"- 対象バケット: {bucket_name}\n"
            "- サービスアカウントに `roles/storage.admin` または少なくとも `roles/storage.objectAdmin` と `roles/storage.legacyBucketReader` を付与してください。"
            f"\n- 詳細: {exc}"
        ) from exc
    except NotFound:
        bucket = storage_client.bucket(bucket_name)
        try:
            normalized_location = (location or "US").upper() if len(location or "") <= 2 else (location or "US")
            if normalized_location.lower() == "global":
                normalized_location = "US"
            storage_client.create_bucket(bucket, location=normalized_location)
        except Conflict as exc:  # pragma: no cover
            raise TranscriptionError(
                f"バケット {bucket_name} を作成できませんでした。別名を --upload-bucket で指定してください。"
            ) from exc
        except Forbidden as exc:
            raise TranscriptionError(
                "Cloud Storage バケットを作成する権限がありません。\n"
                "- プロジェクトに対して `roles/storage.admin` または `roles/storage.bucketCreator` を付与してください。"
                f"\n- 詳細: {exc}"
            ) from exc
        return True


def extract_location_from_recognizer(recognizer_name: str) -> Optional[str]:
    parts = recognizer_name.split("/")
    try:
        index = parts.index("locations")
        return parts[index + 1]
    except (ValueError, IndexError):
        return None


def handle_speech_exception(exc: Exception, recognizer_name: str) -> None:
    location = extract_location_from_recognizer(recognizer_name) or "指定リージョン"
    if isinstance(exc, PermissionDenied):
        raise TranscriptionError(
            "Google Cloud Speech-to-Text v2 API の呼び出しで権限が不足しています。\n"
            f"- 利用中の Recognizer: {recognizer_name}\n"
            "- サービスアカウントに `roles/speech.transcriber` または `roles/cloudspeech.admin` を付与し、"
            "Speech-to-Text API v2 がプロジェクトで有効化されていることを確認してください。"
            f"\n- 詳細: {exc}"
        ) from exc
    if isinstance(exc, NotFound):
        raise TranscriptionError(
            "指定した Recognizer が見つかりません。\n"
            f"- Recognizer: {recognizer_name}\n"
            f"- 対象リージョン ({location}) で `gcloud ml speech recognizers create` を実行してリソースを作成するか、"
            "既存リソースのIDを --recognizer で指定してください。"
            f"\n- 詳細: {exc}"
        ) from exc
    raise TranscriptionError(f"Speech-to-Text API 呼び出しで予期せぬエラーが発生しました: {exc}") from exc


def raise_if_rpc_error(error, recognizer_name: str, context: str) -> None:
    if not error:
        return
    code = getattr(error, "code", 0)
    if not code:
        return
    message = getattr(error, "message", "") or "(詳細情報なし)"
    raise TranscriptionError(
        f"{context} でエラーが返されました。\n"
        f"- Recognizer: {recognizer_name}\n"
        f"- エラーコード: {code}\n"
        f"- 詳細: {message}"
    )


def validate_local_audio(file_path: Path) -> None:
    if not file_path.exists():
        raise TranscriptionError(f"ファイルが見つかりません: {file_path}")
    if not file_path.is_file():
        raise TranscriptionError(f"ファイルではありません: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise TranscriptionError(
            "サポートされていないファイル形式です: "
            f"{file_path.suffix}. 利用可能な拡張子: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )


def resolve_project(args: argparse.Namespace) -> str:
    project = args.project or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
    if project:
        return project

    try:
        from google.auth import default as google_auth_default  # type: ignore
        from google.auth.exceptions import DefaultCredentialsError  # type: ignore
    except ImportError:  # pragma: no cover
        google_auth_default = None
        DefaultCredentialsError = Exception  # type: ignore

    if google_auth_default is not None:
        try:
            _, inferred_project = google_auth_default()
            if inferred_project:
                return inferred_project
        except DefaultCredentialsError:
            pass

    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_path:
        cred_file = Path(cred_path)
        if cred_file.exists():
            try:
                payload = json.loads(cred_file.read_text())
                project = payload.get("project_id") or payload.get("projectId")
                if project:
                    return project
            except (OSError, json.JSONDecodeError) as exc:
                raise TranscriptionError(
                    f"サービスアカウント JSON を読み込めませんでした: {cred_path}: {exc}"
                ) from exc

    raise TranscriptionError(
        "プロジェクト ID が見つかりません。--project 指定、環境変数 GOOGLE_CLOUD_PROJECT 設定、またはプロジェクトを含むサービスアカウント JSON を用意してください。"
    )


def make_client(location: str) -> speech_v2.SpeechClient:
    endpoint = "speech.googleapis.com" if location == "global" else f"{location}-speech.googleapis.com"
    return speech_v2.SpeechClient(client_options=ClientOptions(api_endpoint=endpoint))


def build_recognizer_name(project: str, location: str, recognizer_id: str) -> str:
    if recognizer_id.startswith("projects/"):
        return recognizer_id
    return f"projects/{project}/locations/{location}/recognizers/{recognizer_id}"


def duration_to_millis(duration) -> Optional[int]:
    if not duration:
        return None
    return int(duration.seconds * 1000 + duration.nanos / 1_000_000)


def build_recognition_config(args: argparse.Namespace) -> cloud_speech.RecognitionConfig:
    language_codes = ["auto"] if args.auto_language else args.language_codes

    features = cloud_speech.RecognitionFeatures(
        profanity_filter=args.profanity_filter,
        enable_word_time_offsets=args.enable_word_time_offsets,
        enable_automatic_punctuation=not args.disable_auto_punctuation,
        enable_spoken_punctuation=args.enable_spoken_punctuation,
        enable_spoken_emojis=args.enable_spoken_emojis,
        max_alternatives=args.max_alternatives,
    )

    adaptation = None
    if args.phrase:
        phrases = []
        for phrase in args.phrase:
            entry = {"value": phrase}
            if args.phrase_boost is not None:
                entry["boost"] = args.phrase_boost
            phrases.append(entry)
        adaptation = cloud_speech.SpeechAdaptation(
            phrase_sets=[
                cloud_speech.SpeechAdaptation.AdaptationPhraseSet(
                    inline_phrase_set=cloud_speech.PhraseSet(phrases=phrases)
                )
            ]
        )

    denoiser = None
    if args.denoise_audio or args.snr_threshold is not None:
        denoiser = cloud_speech.DenoiserConfig(
            denoise_audio=args.denoise_audio, snr_threshold=args.snr_threshold or 0.0
        )

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=language_codes,
        model=args.model,
        features=features,
        adaptation=adaptation,
        denoiser_config=denoiser,
    )

    if args.show_config:
        json_config = cloud_speech.RecognitionConfig.to_json(config)
        print(json.dumps(json.loads(json_config), ensure_ascii=False, indent=2), file=sys.stderr)

    return config


def read_file(path: Path) -> bytes:
    return path.read_bytes()


def choose_mode(args: argparse.Namespace, source: str, local_path: Optional[Path]) -> str:
    if args.mode != "auto":
        return args.mode
    if is_gcs_uri(source):
        return "batch"
    if not local_path:
        return "batch"
    size = local_path.stat().st_size
    return "sync" if size <= args.sync_max_bytes else "batch"


def upload_to_gcs(local_path: Path, bucket: str, prefix: str, storage_client) -> str:
    if storage_client is None:  # pragma: no cover
        raise TranscriptionError("Cloud Storage クライアントが初期化されていません。")

    bucket_obj = storage_client.bucket(bucket)
    timestamp = int(time.time())
    trimmed_prefix = prefix.strip("/")
    object_name = f"{trimmed_prefix}/{timestamp}_{local_path.name}" if trimmed_prefix else f"{timestamp}_{local_path.name}"
    blob = bucket_obj.blob(object_name)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket}/{object_name}"


def recognize_sync(
    client: speech_v2.SpeechClient,
    recognizer_name: str,
    config: cloud_speech.RecognitionConfig,
    source: str,
    local_path: Optional[Path],
) -> TranscriptionResult:
    if local_path:
        content = read_file(local_path)
        request = cloud_speech.RecognizeRequest(
            recognizer=recognizer_name,
            config=config,
            content=content,
        )
    else:
        request = cloud_speech.RecognizeRequest(
            recognizer=recognizer_name,
            config=config,
            uri=source,
        )

    try:
        response = client.recognize(request=request)
    except (PermissionDenied, NotFound) as exc:
        handle_speech_exception(exc, recognizer_name)

    raise_if_rpc_error(getattr(response, "error", None), recognizer_name, "同期認識")

    segments: List[Segment] = []
    words: List[Word] = []
    transcript_parts: List[str] = []
    language = None
    last_end_ms = 0
    cumulative_transcript = ""
    cumulative_word_count = 0

    for result in response.results:
        if not result.alternatives:
            continue
        top_alt = result.alternatives[0]
        language = result.language_code or language
        end_ms = duration_to_millis(getattr(result, "result_end_offset", None))
        start_ms = last_end_ms if end_ms is not None else None
        new_transcript = top_alt.transcript or ""
        addition = new_transcript
        if cumulative_transcript and addition.startswith(cumulative_transcript):
            addition = addition[len(cumulative_transcript) :]
        elif cumulative_transcript and cumulative_transcript.endswith(addition):
            addition = ""
        addition = addition.lstrip("\n\r ")
        addition = addition.rstrip()
        if addition:
            transcript_parts.append(addition)
            segments.append(
                Segment(
                    transcript=addition,
                    confidence=getattr(top_alt, "confidence", None),
                    start_ms=start_ms,
                    end_ms=end_ms,
                )
            )
            last_end_ms = end_ms or last_end_ms
        else:
            last_end_ms = end_ms or last_end_ms

        cumulative_transcript = new_transcript or cumulative_transcript

        current_words = list(getattr(top_alt, "words", []))
        if current_words:
            start_index = min(cumulative_word_count, len(current_words))
            for word_info in current_words[start_index:]:
                words.append(
                    Word(
                        word=word_info.word,
                        start_ms=duration_to_millis(getattr(word_info, "start_offset", None)),
                        end_ms=duration_to_millis(getattr(word_info, "end_offset", None)),
                        speaker_label=getattr(word_info, "speaker_label", None) or None,
                    )
                )
            cumulative_word_count = len(current_words)

    transcript = "\n".join(part.strip() for part in transcript_parts if part.strip())
    return TranscriptionResult(
        source=source,
        transcript=transcript,
        language_code=language,
        model=config.model,
        mode="sync",
        segments=segments,
        words=words,
    )


def stream_requests(
    audio_path: Path,
    recognizer_name: str,
    config: cloud_speech.RecognitionConfig,
    chunk_size: int,
    interim: bool,
) -> Iterable[cloud_speech.StreamingRecognizeRequest]:
    streaming_config = cloud_speech.StreamingRecognitionConfig(
        config=config,
        streaming_features=cloud_speech.StreamingRecognitionFeatures(
            interim_results=interim,
            enable_voice_activity_events=False,
        ),
    )
    yield cloud_speech.StreamingRecognizeRequest(
        recognizer=recognizer_name, streaming_config=streaming_config
    )

    with audio_path.open("rb") as audio_file:
        while True:
            chunk = audio_file.read(chunk_size)
            if not chunk:
                break
            yield cloud_speech.StreamingRecognizeRequest(audio=chunk)


def recognize_streaming(
    client: speech_v2.SpeechClient,
    recognizer_name: str,
    config: cloud_speech.RecognitionConfig,
    audio_path: Path,
    chunk_size: int,
    interim: bool,
) -> TranscriptionResult:
    try:
        responses = client.streaming_recognize(
            requests=stream_requests(audio_path, recognizer_name, config, chunk_size, interim)
        )
    except (PermissionDenied, NotFound) as exc:
        handle_speech_exception(exc, recognizer_name)

    segments: List[Segment] = []
    words: List[Word] = []
    transcript_parts: List[str] = []
    language = None
    last_end_ms = 0

    cumulative_transcript = ""
    cumulative_word_count = 0

    try:
        for response in responses:
            raise_if_rpc_error(getattr(response, "error", None), recognizer_name, "ストリーミング認識")
            for result in response.results:
                if not result.alternatives:
                    continue
                top_alt = result.alternatives[0]
                if result.is_final:
                    language = result.language_code or language
                    end_ms = duration_to_millis(getattr(result, "result_end_offset", None))
                    start_ms = last_end_ms if end_ms is not None else None
                    new_transcript = top_alt.transcript or ""
                    addition = new_transcript
                    if cumulative_transcript and addition.startswith(cumulative_transcript):
                        addition = addition[len(cumulative_transcript) :]
                    elif cumulative_transcript and cumulative_transcript.endswith(addition):
                        addition = ""
                    addition = addition.lstrip("\n\r ")
                    addition = addition.rstrip()
                    if addition:
                        transcript_parts.append(addition)
                        segments.append(
                            Segment(
                                transcript=addition,
                                confidence=getattr(top_alt, "confidence", None),
                                start_ms=start_ms,
                                end_ms=end_ms,
                            )
                        )
                        last_end_ms = end_ms or last_end_ms
                    else:
                        last_end_ms = end_ms or last_end_ms

                    cumulative_transcript = new_transcript or cumulative_transcript

                    current_words = list(getattr(top_alt, "words", []))
                    if current_words:
                        start_index = min(cumulative_word_count, len(current_words))
                        for word_info in current_words[start_index:]:
                            words.append(
                                Word(
                                    word=word_info.word,
                                    start_ms=duration_to_millis(getattr(word_info, "start_offset", None)),
                                    end_ms=duration_to_millis(getattr(word_info, "end_offset", None)),
                                    speaker_label=getattr(word_info, "speaker_label", None) or None,
                                )
                            )
                        cumulative_word_count = len(current_words)
                elif interim:
                    print(f"[interim] {top_alt.transcript}", file=sys.stderr)
    except (PermissionDenied, NotFound) as exc:
        handle_speech_exception(exc, recognizer_name)

    transcript = "\n".join(part.strip() for part in transcript_parts if part.strip())
    return TranscriptionResult(
        source=str(audio_path),
        transcript=transcript,
        language_code=language,
        model=config.model,
        mode="streaming",
        segments=segments,
        words=words,
    )


def recognize_batch(
    client: speech_v2.SpeechClient,
    recognizer_name: str,
    config: cloud_speech.RecognitionConfig,
    source_uri: str,
    timeout: int,
) -> TranscriptionResult:
    files = [cloud_speech.BatchRecognizeFileMetadata(uri=source_uri)]

    request = cloud_speech.BatchRecognizeRequest(
        recognizer=recognizer_name,
        config=config,
        files=files,
        recognition_output_config=cloud_speech.RecognitionOutputConfig(
            inline_response_config=cloud_speech.InlineOutputConfig()
        ),
    )

    try:
        operation = client.batch_recognize(request=request)
    except (PermissionDenied, NotFound) as exc:
        handle_speech_exception(exc, recognizer_name)

    try:
        response = operation.result(timeout=timeout)
    except (PermissionDenied, NotFound) as exc:
        handle_speech_exception(exc, recognizer_name)

    if not response.results:
        return TranscriptionResult(
            source=source_uri,
            transcript="",
            language_code=None,
            model=config.model,
            mode="batch",
        )

    file_result = next(iter(response.results.values()))
    if getattr(file_result, "error", None):
        error = file_result.error
        if getattr(error, "code", 0):
            message = getattr(error, "message", "") or "(詳細情報なし)"
            raise TranscriptionError(
                "Batch Recognize でエラーが返されました。\n"
                f"- Recognizer: {recognizer_name}\n"
                f"- 対象 URI: {source_uri}\n"
                f"- エラーコード: {error.code}\n"
                f"- 詳細: {message}"
            )

    inline_result = file_result.inline_result
    if inline_result is None:
        raise TranscriptionError(
            "Batch Recognize から結果が得られませんでした。Cloud Logging にエラーが出ていないか確認してください。"
        )

    transcript_proto = inline_result.transcript

    segments: List[Segment] = []
    words: List[Word] = []
    transcript_parts: List[str] = []
    language = None
    last_end_ms = 0

    cumulative_transcript = ""
    cumulative_word_count = 0

    for result in transcript_proto.results:
        if not result.alternatives:
            continue
        top_alt = result.alternatives[0]
        language = result.language_code or language
        end_ms = duration_to_millis(getattr(result, "result_end_offset", None))
        start_ms = last_end_ms if end_ms is not None else None
        new_transcript = top_alt.transcript or ""
        addition = new_transcript
        if cumulative_transcript and addition.startswith(cumulative_transcript):
            addition = addition[len(cumulative_transcript) :]
        elif cumulative_transcript and cumulative_transcript.endswith(addition):
            addition = ""
        addition = addition.lstrip("\n\r ")
        addition = addition.rstrip()
        if addition:
            transcript_parts.append(addition)
            segments.append(
                Segment(
                    transcript=addition,
                    confidence=getattr(top_alt, "confidence", None),
                    start_ms=start_ms,
                    end_ms=end_ms,
                )
            )
            last_end_ms = end_ms or last_end_ms
        else:
            last_end_ms = end_ms or last_end_ms

        cumulative_transcript = new_transcript or cumulative_transcript

        current_words = list(getattr(top_alt, "words", []))
        if current_words:
            start_index = min(cumulative_word_count, len(current_words))
            for word_info in current_words[start_index:]:
                words.append(
                    Word(
                        word=word_info.word,
                        start_ms=duration_to_millis(getattr(word_info, "start_offset", None)),
                        end_ms=duration_to_millis(getattr(word_info, "end_offset", None)),
                        speaker_label=getattr(word_info, "speaker_label", None) or None,
                    )
                )
            cumulative_word_count = len(current_words)

    transcript = "\n".join(part.strip() for part in transcript_parts if part.strip())
    return TranscriptionResult(
        source=source_uri,
        transcript=transcript,
        language_code=language,
        model=config.model,
        mode="batch",
    )


def ensure_output_dir(path: Optional[Path]) -> None:
    if path is None:
        return
    path.mkdir(parents=True, exist_ok=True)


def render_result(result: TranscriptionResult, fmt: str) -> str:
    if fmt == "text":
        lines = []
        if result.transcript:
            lines.append(result.transcript.strip())
        return "\n".join(lines)

    payload = {
        "source": result.source,
        "mode": result.mode,
        "model": result.model,
        "language_code": result.language_code,
        "transcript": result.transcript,
        "segments": [
            {
                "transcript": segment.transcript,
                "confidence": segment.confidence,
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
            }
            for segment in result.segments
        ],
        "words": [
            {
                "word": word.word,
                "start_ms": word.start_ms,
                "end_ms": word.end_ms,
                "speaker_label": word.speaker_label,
            }
            for word in result.words
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def write_output(content: str, path: Path) -> None:
    path.write_text(content + "\n", encoding="utf-8")


def preprocess_required(args: argparse.Namespace) -> bool:
    return bool(
        args.preprocess_normalize
        or args.preprocess_gain_db != 0
        or args.preprocess_denoise
    )


def apply_noise_reduction(
    audio: "AudioSegment",
    *,
    reduce_amount: float,
    stationary: bool,
) -> "AudioSegment":
    if nr is None or np is None:
        print(
            "警告: noisereduce または numpy が利用できないため、ノイズ除去をスキップします。",
            file=sys.stderr,
        )
        return audio

    print(
        f"ノイズ除去を適用: 強度 {reduce_amount:.2f}, 定常ノイズモード {stationary}",
        file=sys.stderr,
    )

    samples = np.array(audio.get_array_of_samples())
    channels = audio.channels

    if channels > 1:
        samples = samples.reshape((-1, channels))
        samples = samples.mean(axis=1)

    samples = samples.astype(np.float32)
    max_int_value = float(1 << (8 * audio.sample_width - 1))
    if max_int_value == 0:
        return audio
    samples /= max_int_value

    sample_rate = audio.frame_rate
    noise_duration = min(0.5, len(samples) / sample_rate / 2)
    noise_len = int(noise_duration * sample_rate)

    if noise_len > 0:
        noise_clip = samples[:noise_len]
        reduced = nr.reduce_noise(
            y=samples,
            sr=sample_rate,
            y_noise=noise_clip,
            stationary=stationary,
            prop_decrease=reduce_amount,
        )
    else:
        reduced = nr.reduce_noise(
            y=samples,
            sr=sample_rate,
            stationary=stationary,
            prop_decrease=reduce_amount,
        )

    reduced = np.clip(reduced, -1.0, 1.0)
    reduced_int16 = (reduced * 32767).astype(np.int16)

    return AudioSegment(
        reduced_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )


def preprocess_local_audio(
    file_path: Path,
    *,
    normalize: bool,
    target_dbfs: float,
    gain_db: float,
    denoise: bool,
    noise_reduce_amount: float,
    stationary_noise: bool,
) -> Path:
    if AudioSegment is None:
        raise TranscriptionError(
            "音声前処理を有効化するには pydub が必要です。`uv add pydub noisereduce` 等でインストールしてください。"
        )

    print(f"前処理: {file_path}", file=sys.stderr)
    audio = AudioSegment.from_file(str(file_path))

    if denoise:
        reduce_amount = max(0.0, min(1.0, noise_reduce_amount))
        audio = apply_noise_reduction(
            audio,
            reduce_amount=reduce_amount,
            stationary=stationary_noise,
        )

    if gain_db != 0:
        print(f"ゲイン調整: {gain_db:+.1f} dB", file=sys.stderr)
        audio = audio + gain_db

    if normalize:
        if audio.dBFS == float("-inf"):
            print("警告: 無音のためノーマライズをスキップします。", file=sys.stderr)
        else:
            change_in_dbfs = target_dbfs - audio.dBFS
            print(
                f"ノーマライズ: {audio.dBFS:.1f} → {target_dbfs:.1f} dBFS (変化 {change_in_dbfs:+.1f} dB)",
                file=sys.stderr,
            )
            audio = audio.apply_gain(change_in_dbfs)

    processed_path = file_path.with_name(f"{file_path.stem}_preprocessed.wav")
    audio.export(str(processed_path), format="wav")
    print(f"前処理済みファイルを出力: {processed_path}", file=sys.stderr)

    return processed_path


def main() -> int:
    args = parse_args()

    if args.output and len(args.audio) > 1:
        raise TranscriptionError("複数ファイル処理時は --output ではなく --output-dir を使用してください。")

    project = resolve_project(args)
    client = make_client(args.location)
    recognizer_name = build_recognizer_name(project, args.location, args.recognizer)
    config = build_recognition_config(args)

    ensure_output_dir(args.output_dir)

    uploaded_uris: List[str] = []
    storage_client = None
    default_bucket_name: Optional[str] = None
    bucket_notices = set()

    for audio in args.audio:
        original_local_path: Optional[Path] = None
        effective_local_path: Optional[Path] = None
        source_uri: Optional[str] = None
        cleanup_paths: List[Path] = []

        try:
            if is_gcs_uri(audio):
                source_uri = audio
            else:
                original_local_path = Path(audio)
                validate_local_audio(original_local_path)
                source_uri = str(original_local_path)
                effective_local_path = original_local_path

                if preprocess_required(args):
                    processed_path = preprocess_local_audio(
                        original_local_path,
                        normalize=args.preprocess_normalize,
                        target_dbfs=args.preprocess_target_dbfs,
                        gain_db=args.preprocess_gain_db,
                        denoise=args.preprocess_denoise,
                        noise_reduce_amount=args.preprocess_noise_reduce_amount,
                        stationary_noise=args.preprocess_stationary_noise,
                    )
                    cleanup_paths.append(processed_path)
                    effective_local_path = processed_path
                    source_uri = str(processed_path)

            if source_uri is None:
                raise TranscriptionError("音声ソースを判別できませんでした。")

            mode = choose_mode(args, source_uri, effective_local_path)

            if mode == "streaming" and source_uri.startswith("gs://"):
                raise TranscriptionError("ストリーミング認識では gs:// URI を直接指定できません。ローカルにダウンロードしてください。")

            if mode == "batch":
                if storage_client is None:
                    try:
                        from google.cloud import storage
                    except ImportError as exc:  # pragma: no cover
                        raise TranscriptionError(
                            "google-cloud-storage がインストールされていません。uv sync を実行してください。"
                        ) from exc
                    storage_client = storage.Client(project=project)

                if args.upload_bucket:
                    target_bucket = args.upload_bucket
                else:
                    if default_bucket_name is None:
                        default_bucket_name = derive_default_bucket_name(project, args.location)
                    target_bucket = default_bucket_name

                bucket_created = ensure_bucket_exists(storage_client, target_bucket, args.location)
                if target_bucket not in bucket_notices:
                    message = (
                        f"Batch処理用にバケット {target_bucket} を新規作成しました。"
                        if bucket_created
                        else f"Batch処理用にバケット {target_bucket} を使用します。"
                    )
                    print(message, file=sys.stderr)
                    bucket_notices.add(target_bucket)

                if not is_gcs_uri(source_uri):
                    if effective_local_path is None:  # pragma: no cover
                        raise TranscriptionError("ローカルファイルのパスが特定できませんでした。")
                    uploaded_uri = upload_to_gcs(effective_local_path, target_bucket, args.upload_prefix, storage_client)
                    uploaded_uris.append(uploaded_uri)
                    source_uri = uploaded_uri

                result = recognize_batch(
                    client,
                    recognizer_name,
                    config,
                    source_uri,
                    timeout=args.batch_timeout,
                )
            elif mode == "sync":
                if is_gcs_uri(source_uri):
                    result = recognize_sync(client, recognizer_name, config, source_uri, None)
                else:
                    if effective_local_path is None:
                        raise TranscriptionError("ローカルファイルのパスが特定できませんでした。")
                    result = recognize_sync(
                        client,
                        recognizer_name,
                        config,
                        source_uri,
                        effective_local_path,
                    )
            elif mode == "streaming":
                if not effective_local_path:
                    raise TranscriptionError("ストリーミングはローカルファイルのみ対応しています。")
                result = recognize_streaming(
                    client,
                    recognizer_name,
                    config,
                    effective_local_path,
                    chunk_size=args.chunk_size,
                    interim=args.interim_results,
                )
            else:
                raise TranscriptionError(f"未知のモードです: {mode}")

            output_text = render_result(result, args.format)
            if not result.transcript.strip():
                print(
                    "警告: 文字起こし結果が空でした。音声が無音か、API がエラーを返した可能性があります。",
                    file=sys.stderr,
                )

            if args.output:
                write_output(output_text, args.output)
            elif args.output_dir:
                suffix = ".json" if args.format == "json" else ".txt"
                target = args.output_dir / (Path(audio).stem + suffix)
                write_output(output_text, target)
            else:
                print(f"=== {audio} ({result.mode}) ===")
                if output_text:
                    print(output_text)

                if original_local_path is not None:
                    default_suffix = ".json" if args.format == "json" else ".txt"
                    default_path = original_local_path.with_suffix(default_suffix)
                    write_output(output_text, default_path)
                    print(f"[saved] {default_path}")
        finally:
            for temp_path in cleanup_paths:
                if original_local_path and temp_path == original_local_path:
                    continue
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    if uploaded_uris:
        joined = "\n".join(uploaded_uris)
        print("アップロード済みの Cloud Storage URI:", file=sys.stderr)
        print(joined, file=sys.stderr)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TranscriptionError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        raise SystemExit(1)
