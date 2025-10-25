"""
日本語Text-to-Speech実装
複数のTTSエンジンをサポート
"""

import os
from pathlib import Path
from typing import Optional, List
import torch
import torchaudio
import numpy as np
from abc import ABC, abstractmethod
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import re
import soundfile as sf
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel
from huggingface_hub import hf_hub_download
import scipy.signal
import sys


class TTSEngine(ABC):
    """TTSエンジンの基底クラス"""
    
    @abstractmethod
    def generate_speech(self, text: str, output_path: Path, target_duration: Optional[float] = None):
        """音声を生成"""
        pass
    
    @abstractmethod
    def estimate_duration(self, text: str) -> float:
        """テキストから音声の長さを推定"""
        pass


class ESPnetTTS(TTSEngine):
    """ESPnet日本語TTSエンジン"""
    
    def __init__(self, model_tag: str = "kan-bayashi/jsut_full_band_vits_prosody"):
        try:
            from espnet2.bin.tts_inference import Text2Speech
            self.model_tag = model_tag
            self.text2speech = Text2Speech.from_pretrained(
                model_tag=model_tag,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            print(f"ESPnetモデルをロード: {model_tag}")
        except ImportError:
            print("警告: espnet2がインストールされていません")
            print("インストール: pip install espnet espnet_model_zoo")
            raise
    
    def generate_speech(self, text: str, output_path: Path, target_duration: Optional[float] = None):
        """音声を生成"""
        # 音声合成
        with torch.no_grad():
            wav = self.text2speech(text)["wav"]
        
        # 音声の長さを調整（オプション）
        if target_duration is not None:
            current_duration = len(wav) / self.text2speech.fs
            if abs(current_duration - target_duration) > 1.0:  # 1秒以上の差がある場合
                speed_ratio = current_duration / target_duration
                wav = self._adjust_speed(wav, speed_ratio, self.text2speech.fs)
        
        # 保存
        torchaudio.save(
            str(output_path),
            wav.view(1, -1).cpu(),
            self.text2speech.fs
        )
        
        return len(wav) / self.text2speech.fs
    
    def _adjust_speed(self, wav: torch.Tensor, speed_ratio: float, sr: int) -> torch.Tensor:
        """音声の速度を調整（pyrubberbandを使用）"""
        try:
            import pyrubberband as prb
        except ImportError:
            print("警告: pyrubberbandがインストールされていません。高品質な速度調整にはpyrubberbandが必要です。", file=sys.stderr)
            print("インストール: pip install pyrubberband", file=sys.stderr)
            raise

        # wavを確実にnumpy.ndarrayのfloat32型に変換し、[-1, 1]の範囲に正規化
        wav_np = wav.cpu().numpy()
        if np.issubdtype(wav_np.dtype, np.integer):
            wav_np = wav_np.astype(np.float32) / np.iinfo(wav_np.dtype).max
        else:
            wav_np = wav_np.astype(np.float32)
            if np.max(np.abs(wav_np)) > 1.0:
                wav_np = wav_np / np.max(np.abs(wav_np))

        # 速度調整の範囲を制限
        speed_ratio = np.clip(speed_ratio, 0.8, 1.3)
        
        # pyrubberband を使用して速度調整
        adjusted_wav_np = prb.time_stretch(wav_np, sr, rate=1.0 / speed_ratio)
        
        return torch.from_numpy(adjusted_wav_np).to(wav.device)
    
    def estimate_duration(self, text: str) -> float:
        """テキストから音声の長さを推定"""
        # 日本語: 約6文字/秒（平均的な話速）
        chars = len(text)
        return chars / 6.0


class SpeechT5TTS(TTSEngine):
    """SpeechT5 TTSエンジン（多言語対応）"""
    
    def __init__(self):
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        from datasets import load_dataset
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # モデルをロード
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
        
        # スピーカー埋め込みをロード
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(self.device)
        
        print("SpeechT5モデルをロード完了")
    
    def generate_speech(self, text: str, output_path: Path, target_duration: Optional[float] = None):
        """音声を生成"""
        # テキストを処理
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        
        # 音声を生成
        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"],
                self.speaker_embeddings,
                vocoder=self.vocoder
            )
        
        # 速度調整（オプション）
        if target_duration is not None:
            current_duration = len(speech) / 16000
            if abs(current_duration - target_duration) > 1.0:
                speed_ratio = current_duration / target_duration
                speech = self._adjust_speed(speech, speed_ratio, 16000)
        
        # 保存
        torchaudio.save(
            str(output_path),
            speech.unsqueeze(0).cpu(),
            16000
        )
        
        return len(speech) / 16000
    
    def _adjust_speed(self, wav: torch.Tensor, speed_ratio: float, sr: int) -> torch.Tensor:
        """音声の速度を調整"""
        try:
            import pyrubberband as prb
        except ImportError:
            print("警告: pyrubberbandがインストールされていません。高品質な速度調整にはpyrubberbandが必要です。", file=sys.stderr)
            print("インストール: pip install pyrubberband", file=sys.stderr)
            raise

        # wavを確実にnumpy.ndarrayのfloat32型に変換し、[-1, 1]の範囲に正規化
        wav_np = wav.cpu().numpy()
        if np.issubdtype(wav_np.dtype, np.integer):
            wav_np = wav_np.astype(np.float32) / np.iinfo(wav_np.dtype).max
        else:
            wav_np = wav_np.astype(np.float32)
            if np.max(np.abs(wav_np)) > 1.0:
                wav_np = wav_np / np.max(np.abs(wav_np))

        # 速度調整の範囲を制限
        speed_ratio = np.clip(speed_ratio, 0.8, 1.3)
        
        # pyrubberband を使用して速度調整
        adjusted_wav_np = prb.time_stretch(wav_np, sr, rate=1.0 / speed_ratio)
        
        return torch.from_numpy(adjusted_wav_np).to(wav.device)
    
    def estimate_duration(self, text: str) -> float:
        """テキストから音声の長さを推定"""
        words = len(text.split())
        return words / 2.5  # 約2.5単語/秒


class VOICEVOXEngine(TTSEngine):
    """VOICEVOX API連携（ローカルサーバー必要）"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 50021, speaker_id: int = 1):
        import requests
        self.base_url = f"http://{host}:{port}"
        self.speaker_id = speaker_id
        self.session = requests.Session()
        
        try:
            # サーバー接続確認
            response = self.session.get(f"{self.base_url}/speakers")
            response.raise_for_status()
            print(f"VOICEVOXサーバーに接続: {self.base_url}")
        except Exception as e:
            print(f"警告: VOICEVOXサーバーに接続できません: {e}")
            print("VOICEVOXエンジンをインストール・起動してください")
            raise
    
    def generate_speech(self, text: str, output_path: Path, target_duration: Optional[float] = None):
        """音声を生成"""
        import requests
        
        # 音声合成クエリを作成
        query_response = self.session.post(
            f"{self.base_url}/audio_query",
            params={"text": text, "speaker": self.speaker_id}
        )
        query_response.raise_for_status()
        query_data = query_response.json()
        
        # 速度調整（オプション）
        if target_duration is not None:
            estimated_duration = self._calculate_duration(query_data)
            if abs(estimated_duration - target_duration) > 1.0:
                speed_scale = estimated_duration / target_duration
                speed_scale = np.clip(speed_scale, 0.5, 2.0)
                query_data["speedScale"] = speed_scale
        
        # 音声合成
        synthesis_response = self.session.post(
            f"{self.base_url}/synthesis",
            params={"speaker": self.speaker_id},
            json=query_data
        )
        synthesis_response.raise_for_status()
        
        # 保存
        with open(output_path, "wb") as f:
            f.write(synthesis_response.content)
        
        return self._calculate_duration(query_data)
    
    def _calculate_duration(self, query_data: dict) -> float:
        """音声の長さを計算"""
        total_length = sum(
            mora["frame_length"] for accent_phrase in query_data["accent_phrases"]
            for mora in accent_phrase["moras"]
        )
        return total_length / query_data["outputSamplingRate"]
    
    def estimate_duration(self, text: str) -> float:
        """テキストから音声の長さを推定"""
        chars = len(text)
        return chars / 6.0  # 約6文字/秒


class GTTSEngine(TTSEngine):
    """Google Text-to-Speech（簡易版、オフライン不可）"""
    
    def __init__(self, lang: str = "ja"):
        from gtts import gTTS
        self.lang = lang
        print("gTTS エンジンを使用（インターネット接続が必要）")
    
    def generate_speech(self, text: str, output_path: Path, target_duration: Optional[float] = None):
        """音声を生成"""
        from gtts import gTTS
        from pydub import AudioSegment
        
        # 音声を生成
        tts = gTTS(text=text, lang=self.lang, slow=False)
        
        # 一時ファイルに保存
        temp_path = output_path.with_suffix('.mp3.temp')
        tts.save(str(temp_path))
        
        # 速度調整（オプション）
        if target_duration is not None:
            audio = AudioSegment.from_mp3(str(temp_path))
            current_duration = len(audio) / 1000.0
            
            if abs(current_duration - target_duration) > 1.0:
                speed_ratio = current_duration / target_duration
                speed_ratio = np.clip(speed_ratio, 0.7, 1.5)
                
                # 速度変更
                audio = audio._spawn(audio.raw_data, overrides={
                    "frame_rate": int(audio.frame_rate * speed_ratio)
                })
                audio = audio.set_frame_rate(audio.frame_rate)
            
            audio.export(str(output_path), format="mp3")
            temp_path.unlink()
            return len(audio) / 1000.0
        else:
            temp_path.rename(output_path)
            audio = AudioSegment.from_mp3(str(output_path))
            return len(audio) / 1000.0
    
    def estimate_duration(self, text: str) -> float:
        """テキストから音声の長さを推定"""
        chars = len(text)
        return chars / 6.0


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

    def generate_speech(self, text: str, output_path: Path, target_duration: Optional[float] = None):
        """音声を生成"""
        sr, audio = self.model.infer(text=text)

        # 速度調整（オプション）
        if target_duration is not None:
            current_duration = len(audio) / sr
            if abs(current_duration - target_duration) > 1.0:  # 1秒以上の差がある場合
                speed_ratio = current_duration / target_duration
                audio = self._adjust_speed(audio, speed_ratio, sr) # srを削除

        # 保存
        sf.write(str(output_path), audio, samplerate=sr)

        return len(audio) / sr

    def _adjust_speed(self, wav: np.ndarray, speed_ratio: float, sr: int) -> np.ndarray:
        """音声の速度を調整（pyrubberbandを使用）"""
        try:
            import pyrubberband as prb
        except ImportError:
            print("警告: pyrubberbandがインストールされていません。ピッチを維持した速度調整にはpyrubberbandが必要です。", file=sys.stderr)
            print("インストール: pip install pyrubberband", file=sys.stderr)
            raise

        # wavを確実にnumpy.ndarrayのfloat32型に変換し、[-1, 1]の範囲に正規化
        if np.issubdtype(wav.dtype, np.integer):
            wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
        else:
            wav = wav.astype(np.float32)
            if np.max(np.abs(wav)) > 1.0:
                wav = wav / np.max(np.abs(wav))

        # 速度調整の範囲を制限
        speed_ratio = np.clip(speed_ratio, 0.8, 1.3)

        # pyrubberband.time_stretch を使用して速度調整
        adjusted_wav = prb.time_stretch(y=wav, sr=sr, rate=1.0 / speed_ratio)

        return adjusted_wav

    def estimate_duration(self, text: str) -> float:
        """テキストから音声の長さを推定"""
        # 日本語: 約6文字/秒（平均的な話速）
        chars = len(text)
        return chars / 6.0


def get_tts_engine(engine_name: str = "auto", **kwargs) -> TTSEngine:
    """
    TTSエンジンを取得
    
    Args:
        engine_name: エンジン名 ("mms", "espnet", "speecht5", "voicevox", "gtts", "auto", "style-bert-vits2")
        **kwargs: エンジン固有のパラメータ
    
    Returns:
        TTSEngine: 指定されたTTSエンジン
    """
    
    if engine_name == "auto":
        # 優先順位: StyleBertVits2 -> VOICEVOX -> ESPnet -> SpeechT5 -> gTTS
        try:
            return StyleBertVits2TTS(**kwargs)
        except:
            pass
        
        try:
            return VOICEVOXEngine(**kwargs)
        except:
            pass
        
        try:
            return ESPnetTTS(**kwargs)
        except:
            pass
        
        try:
            return SpeechT5TTS(**kwargs)
        except:
            pass
        
        try:
            print("Google TTS (gTTS)にフォールバック...")
            return GTTSEngine(**kwargs)
        except:
            pass
        
        raise RuntimeError("利用可能なTTSエンジンが見つかりません")
    
    elif engine_name == "espnet":
        return ESPnetTTS(**kwargs)
    
    elif engine_name == "speecht5":
        return SpeechT5TTS(**kwargs)
    
    elif engine_name == "voicevox":
        return VOICEVOXEngine(**kwargs)
    
    elif engine_name == "gtts":
        return GTTSEngine(**kwargs)
    
    elif engine_name == "style-bert-vits2":
        return StyleBertVits2TTS(**kwargs)
    
    else:
        raise ValueError(f"未知のエンジン: {engine_name}")


# 使用例
if __name__ == "__main__":
    # エンジンを取得
    engine = get_tts_engine("auto")
    
    # テキスト
    text = "こんにちは。これはテスト音声です。"
    
    # 音声生成
    output_path = Path("test_output.mp3")
    duration = engine.generate_speech(text, output_path, target_duration=3.0)
    
    print(f"音声を生成しました: {output_path}")
    print(f"音声の長さ: {duration:.2f}秒")