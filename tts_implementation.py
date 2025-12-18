"""
日本語Text-to-Speech実装
"""

import os
from pathlib import Path
from typing import Optional
import torch
import numpy as np
from abc import ABC, abstractmethod
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel
from huggingface_hub import hf_hub_download
import soundfile as sf
import sys
import io
from pydub import AudioSegment


class TTSEngine(ABC):
    """TTSエンジンの基底クラス"""
    
    @abstractmethod
    def generate_speech(self, text: str, output_path: Path, target_duration: Optional[float] = None) -> float:
        """音声を生成"""
        pass
    
    @abstractmethod
    def estimate_duration(self, text: str) -> float:
        """テキストから音声の長さを推定"""
        pass


class StyleBertVits2TTS(TTSEngine):
    """Style-Bert-Vits2 TTSエンジン"""

    def __init__(self, device: str = None, **kwargs):
        try:
            # デバイス設定
            self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

            # BERTモデルとトークナイザーのロード
            bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
            bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

            # モデルファイルのダウンロード
            assets_root = Path("model_assets")
            model_file = "jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors"
            config_file = "jvnv-F1-jp/config.json"
            style_file = "jvnv-F1-jp/style_vectors.npy"

            for file in [model_file, config_file, style_file]:
                hf_hub_download("litagin/style_bert_vits2_jvnv", file, local_dir=assets_root)

            # TTSモデルの初期化
            self.model = TTSModel(
                model_path=assets_root / model_file,
                config_path=assets_root / config_file,
                style_vec_path=assets_root / style_file,
                device=self.device,
            )
            print(f"Style-Bert-Vits2モデルをロード完了 (デバイス: {self.device})")

        except ImportError:
            print("警告: style_bert_vits2がインストールされていません")
            print("インストール: pip install style_bert_vits2")
            raise
        except Exception as e:
            print(f"Style-Bert-Vits2モデルのロード中にエラーが発生しました: {e}")
            raise

    def generate_speech(self, text: str, output_path: Path, target_duration: Optional[float] = None) -> float:
        """音声を生成"""
        sr, audio = self.model.infer(text=text, style_weight=0.2, assist_text="落ち着いたナレーション")

        current_duration = len(audio) / sr

        # 速度調整（オプション）
        if target_duration is not None:
            diff = target_duration - current_duration
            if diff < -5 or diff > -1:
                sr, audio = self.model.infer(text=text, style_weight=0.2, assist_text="落ち着いたナレーション", length=(target_duration-4)/current_duration)

        current_duration = len(audio) / sr

        # 空白で調整
        if current_duration < target_duration:
            padding_duration = target_duration - current_duration
            padding_samples = int(padding_duration * sr)
            if padding_samples > 0:
                audio = np.concatenate((audio, np.zeros(padding_samples, dtype=audio.dtype)))

        # 保存
        if str(output_path).endswith('.mp3'):
            # WAVとしてメモリに書き出し
            buffer = io.BytesIO()
            sf.write(buffer, audio, sr, format='WAV')
            buffer.seek(0)
            
            # MP3に変換して保存
            AudioSegment.from_wav(buffer).export(str(output_path), format="mp3")
        else:
            sf.write(str(output_path), audio, samplerate=sr)

        return len(audio) / sr

    def estimate_duration(self, text: str) -> float:
        """テキストから音声の長さを推定"""
        # 日本語: 約6文字/秒（平均的な話速）
        chars = len(text)
        return chars / 6.0


def get_tts_engine(**kwargs) -> TTSEngine:
    """
    TTSエンジンを取得
    
    Args:
        **kwargs: エンジン固有のパラメータ
    
    Returns:
        TTSEngine: StyleBertVits2TTSエンジン
    """
    try:
        return StyleBertVits2TTS(**kwargs)
    except Exception as e:
        print(f"TTSエンジンの初期化に失敗しました: {e}", file=sys.stderr)
        raise RuntimeError("TTSエンジンの初期化に失敗しました。") from e


# 使用例
if __name__ == "__main__":
    # エンジンを取得
    engine = get_tts_engine()
    
    # テキスト
    text = "こんにちは。これはテスト音声です。"
    
    # 音声生成
    output_path = Path("test_output.mp3")
    duration = engine.generate_speech(text, output_path, target_duration=3.0)
    
    print(f"音声を生成しました: {output_path}")
    print(f"音声の長さ: {duration:.2f}秒")