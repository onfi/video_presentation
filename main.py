#!/usr/bin/env python3
"""
プレゼンテーション動画自動生成システム
Text to SpeechとMarpを使用して、日本語プレゼンテーション動画を生成
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Optional
import re
from google import genai
import time

import torch
import soundfile as sf

# 設定とプロンプトをインポート
from config import JAPANESE_READING_SPEED, PromptTemplates

def _call_genai_with_retry(prompt: str, model: str = "gemini-3-flash-preview", max_retries: int = 10) -> str:
    """Call GenAI API with retry logic for overloaded errors.
    
    Args:
        prompt: The prompt to send
        model: Model name to use
        max_retries: Maximum number of retries (default: 10)
        
    Returns:
        API response text
    """
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model, contents=prompt
            )
            time.sleep(1)
            return response.text
        except Exception as e:
            error_str = str(e)
            # "The model is overloaded" エラーの場合のみリトライ
            if 'The model is overloaded' in error_str:
                if attempt < max_retries - 1:
                    print(f"モデルが過負荷状態です。77秒待機してリトライします... (試行 {attempt + 1}/{max_retries})")
                    time.sleep(77)
                else:
                    print(f"最大リトライ回数({max_retries})に達しました")
                    raise
            else:
                # その他のエラーは即座に再送出
                raise
    
    raise Exception(f"リトライ回数上限({max_retries})に達しました")

def _call_genai(prompt: str, model: str = "gemini-3-flash-preview") -> str:
    """Call the available GenAI API and return text.

    Uses new client API if available (`client.models.generate_content`),
    otherwise uses older `generate_text`/`chat` functions.
    """
    return _call_genai_with_retry(prompt, model)

def _call_genai_structured(prompt: str, response_schema: dict, model: str = "gemini-3-flash-preview", max_retries: int = 10) -> dict:
    """Call Gemini API with structured output.
    
    Args:
        prompt: The prompt to send
        response_schema: JSON schema defining the expected response structure
        model: Model name to use
        max_retries: Maximum number of retries for overloaded errors (default: 10)
        
    Returns:
        Parsed JSON response as a dictionary
    """
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_schema
                }
            )
            time.sleep(1)
            return json.loads(response.text)
        except Exception as e:
            error_str = str(e)
            # "The model is overloaded" エラーの場合のみリトライ
            if 'The model is overloaded' in error_str:
                if attempt < max_retries - 1:
                    print(f"モデルが過負荷状態です。77秒待機してリトライします... (試行 {attempt + 1}/{max_retries})")
                    time.sleep(77)
                else:
                    print(f"最大リトライ回数({max_retries})に達しました")
                    raise
            else:
                # その他のエラーは即座に再送出
                raise
    
    raise Exception(f"リトライ回数上限({max_retries})に達しました")

class PresentationGenerator:
    """プレゼンテーション動画生成のメインクラス"""
    
    def __init__(self, project_name: str, base_dir: str = "./outputs"):
        self.project_name = project_name
        self.base_dir = Path(base_dir)
        self.project_dir = self.base_dir / project_name
        
        # ディレクトリ構造を作成
        self.project_dir.mkdir(parents=True, exist_ok=True)
        (self.project_dir / "slides").mkdir(exist_ok=True)
        (self.project_dir / "narration").mkdir(exist_ok=True)
        (self.project_dir / "audio").mkdir(exist_ok=True)
        (self.project_dir / "video").mkdir(exist_ok=True)
    
    def _get_slide_path(self, index: int, ext: str = "png") -> Path:
        """スライドファイルのパスを取得"""
        return self.project_dir / "slides" / f"{index:03d}.{ext}"
    
    def _get_narration_path(self, index: int) -> Path:
        """ナレーションテキストファイルのパスを取得"""
        return self.project_dir / "narration" / f"{index:03d}.txt"
    
    def _get_audio_path(self, index: int) -> Path:
        """音声ファイルのパスを取得"""
        return self.project_dir / "audio" / f"slide{index:03d}.mp3"
    
    def _get_video_path(self, index: int) -> Path:
        """スライド動画ファイルのパスを取得"""
        return self.project_dir / "video" / f"slide{index:03d}.mp4"


class SlideMarkdownGenerator:
    """スライド用Markdownを生成"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        os.environ["GEMINI_API_KEY"] = api_key
    
    def generate(self, content_file: Path, num_slides: Optional[int] = None, 
                 time: int = 180, project_name: str = None) -> str:
        """プレゼンテーション内容からスライドMarkdownを生成"""
        
        # コンテンツファイルを読み込み
        with open(content_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # プロンプト生成時点でのスライド数を保持
        prompt_num_slides = num_slides

        # 計算用のスライド数を確定
        if num_slides is None:
            calc_num_slides = max(3, time // 60)  # 1分あたり1枚程度
        else:
            calc_num_slides = num_slides
        
        # 1スライドあたりの秒数と文字数を計算
        time_per_slide = time / calc_num_slides
        chars_per_slide = int(time_per_slide * JAPANESE_READING_SPEED / 60)
        
        # プロンプトとスキーマを取得
        prompt = PromptTemplates.slide_generation(content, prompt_num_slides, chars_per_slide, time_per_slide)
        response_schema = PromptTemplates.slide_schema(calc_num_slides)
        
        try:
            result = _call_genai_structured(prompt, response_schema)
            markdown = result['markdown']
                
        except Exception as e:
            print("エラー: Generative API 呼び出しに失敗しました:")
            print(str(e))
            print("ヒント: Google Cloud の認証（Application Default Credentials）やモデルアクセスを確認してください。")
            sys.exit(1)
        
        # プロジェクト名を決定
        if project_name is None:
            project_name = content_file.stem
        
        # 出力
        output_dir = Path("./outputs") / project_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "slides.md"
        
        # 既存のslides.mdをクリーンアップ
        if output_path.exists():
            output_path.unlink()
            print(f"既存のslides.mdを削除しました")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        print(f"スライドMarkdownを生成しました: {output_path}")
        return project_name


class NarrationTextGenerator:
    """ナレーション用テキストを生成"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        os.environ["GEMINI_API_KEY"] = api_key
    
    def generate(self, project_name: str, total_time: int = 180):
        """スライドMarkdownから各スライドのナレーションテキストを生成"""
        
        gen = PresentationGenerator(project_name)
        slides_md_path = gen.project_dir / "slides.md"
        
        # narrationディレクトリをクリーンアップ
        narration_dir = gen.project_dir / "narration"
        if narration_dir.exists():
            import shutil
            shutil.rmtree(narration_dir)
            print(f"既存のnarrationディレクトリを削除しました")
        narration_dir.mkdir(parents=True, exist_ok=True)
        
        # Markdownを読み込み
        with open(slides_md_path, 'r', encoding='utf-8') as f:
            markdown = f.read()
        
        # Front Matterを除去（先頭の --- ... --- の部分）
        if markdown.strip().startswith('---'):
            parts = markdown.split('---', 2)
            if len(parts) >= 3:
                markdown = parts[2]
        
        # スライドごとに分割（Marp形式: `---`で区切られる）
        slides = re.split(r'^---$', markdown, flags=re.MULTILINE)
        slides = [s.strip() for s in slides if s.strip()]
        
        print(f"{len(slides)}枚のスライドを検出しました")
        
        # 1スライドあたりの秒数と文字数を計算
        time_per_slide = total_time / len(slides)
        chars_per_slide = int(time_per_slide * JAPANESE_READING_SPEED / 60)
        
        # 全スライドのナレーションを一度に生成
        slides_text = ""
        for i, slide in enumerate(slides):
            slides_text += f"\n--- スライド{i+1} ---\n{slide}\n"
        
        # プロンプトとスキーマを取得
        prompt = PromptTemplates.narration_generation(slides_text, len(slides), chars_per_slide, time_per_slide)
        response_schema = PromptTemplates.narration_schema(len(slides), chars_per_slide, time_per_slide)
        
        try:
            result = _call_genai_structured(prompt, response_schema)
            narrations = result['narrations']
            
            # スライド番号の検証
            slide_numbers = [n['slide'] for n in narrations]
            expected_numbers = list(range(1, len(slides) + 1))
            missing_slides = set(expected_numbers) - set(slide_numbers)
            
            if missing_slides:
                print(f"警告: 以下のスライドのナレーションが生成されませんでした: {sorted(missing_slides)}")
            
            # 各ナレーションをファイルに保存
            for narration_item in narrations:
                i = narration_item['slide'] - 1
                if i < 0 or i >= len(slides):
                    print(f"警告: 無効なスライド番号 {narration_item['slide']} をスキップします")
                    continue
                    
                text = narration_item['text']
                char_count = len(text)
                
                # 文字数の警告
                if abs(char_count - chars_per_slide) > 20:
                    print(f"警告: スライド {i:03d} の文字数が目標から大きく外れています（{char_count}文字、目標{chars_per_slide}文字）")
                
                output_path = gen._get_narration_path(i)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                print(f"ナレーション {i:03d} を生成しました（{char_count}文字）")
                
        except Exception as e:
            print("エラー: Generative API 呼び出しまたはJSON解析に失敗しました:")
            print(str(e))
            print("ヒント: Google Cloud の認証（Application Default Credentials）やモデルアクセスを確認してください。")
            sys.exit(1)


class SlideImageGenerator:
    """スライド画像を生成"""
    
    def generate(self, project_name: str, theme: str = "default"):
        """Marp CLIを使用してスライド画像を生成"""
        
        gen = PresentationGenerator(project_name)
        slides_md_path = gen.project_dir / "slides.md"
        output_dir = gen.project_dir / "slides"
        
        # slidesディレクトリをクリーンアップ
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
            print(f"既存のslidesディレクトリを削除しました")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Marp CLIコマンド実行
        # 出力ファイル名を指定（Marpが自動的に.001.png, .002.pngなどを追加）
        output_path = output_dir / "slide"
        cmd = [
            "marp",
            str(slides_md_path),
            "--theme", theme,
            "--images", "png",
            "--output", str(output_path) + ".png",
            "--image-scale", "1.5",
            "--allow-local-files"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # ファイル名を変更: slide.001.png -> 001.png
            for slide_file in sorted(output_dir.glob("slide.*.png")):
                new_name = slide_file.name.replace("slide.", "")
                slide_file.rename(output_dir / new_name)
            
            print(f"スライド画像を生成しました: {output_dir}")
        except subprocess.CalledProcessError as e:
            print(f"エラー: Marp CLIの実行に失敗しました: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print("エラー: Marp CLIが見つかりません。インストールしてください。")
            print("npm install -g @marp-team/marp-cli")
            sys.exit(1)


class NarrationAudioGenerator:
    """ナレーション音声を生成"""
    
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: エンジン固有のパラメータ
        """
        self.engine_kwargs = kwargs
        self.engine = None
    
    def _load_engine(self):
        """TTSエンジンをロード"""
        if self.engine is None:
            from tts_implementation import get_tts_engine
            print(f"TTSエンジンをロード中...")
            self.engine = get_tts_engine(**self.engine_kwargs)
    
    def generate(self, project_name: str, total_time: int = 180):
        """ナレーション音声を生成"""
        
        self._load_engine()
        gen = PresentationGenerator(project_name)
        
        # audioディレクトリをクリーンアップ
        audio_dir = gen.project_dir / "audio"
        if audio_dir.exists():
            import shutil
            shutil.rmtree(audio_dir)
            print(f"既存のaudioディレクトリを削除しました")
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        # ナレーションファイルを取得
        narration_files = sorted(gen.project_dir.glob("narration/*.txt"))
        
        if not narration_files:
            print("エラー: ナレーションファイルが見つかりません")
            sys.exit(1)
        
        # 各ナレーションの文字数を取得
        text_lengths = []
        narrations = []
        for nf in narration_files:
            with open(nf, 'r', encoding='utf-8') as f:
                text = f.read()
                narrations.append(text)
                text_lengths.append(len(text))
        
        # 文字数に応じて時間を割り振る
        total_chars = sum(text_lengths)
        durations = [(length / total_chars) * total_time for length in text_lengths]
        
        print(f"{len(narrations)}個の音声を生成します")
        
        # 各ナレーションの音声を生成
        for i, (text, target_duration) in enumerate(zip(narrations, durations)):
            print(f"音声 {i:03d} を生成中... (目標時間: {target_duration:.1f}秒)")
            
            output_path = gen._get_audio_path(i)
            
            try:
                # 音声を生成
                actual_duration = self.engine.generate_speech(
                    text, 
                    output_path, 
                    target_duration=target_duration
                )
                print(f"音声 {i:03d} を生成しました: {output_path} (実際の長さ: {actual_duration:.1f}秒)")
            except Exception as e:
                print(f"エラー: 音声 {i:03d} の生成に失敗しました: {e}")
                raise


class SlideVideoGenerator:
    """スライドごとの動画を生成"""
    
    def generate(self, project_name: str):
        """スライド画像と音声を結合して動画を生成"""
        gen = PresentationGenerator(project_name)
        
        # videoディレクトリをクリーンアップ
        video_dir = gen.project_dir / "video"
        if video_dir.exists():
            import shutil
            shutil.rmtree(video_dir)
            print(f"既存のvideoディレクトリを削除しました")
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # スライド画像と音声ファイルを取得
        slide_images = sorted(gen.project_dir.glob("slides/*.png"))
        audio_files = sorted(gen.project_dir.glob("audio/*.mp3"))
        
        if len(slide_images) != len(audio_files):
            print(f"警告: スライド数({len(slide_images)})と音声数({len(audio_files)})が一致しません")
        
        # 各スライドの動画を生成（ffmpegを直接使用で高速化）
        for i, (img, audio) in enumerate(zip(slide_images, audio_files)):
            print(f"動画 {i:03d} を生成中...")
            
            output_path = gen._get_video_path(i)
            
            # ffmpegコマンドで高速生成
            # -loop 1: 画像をループ
            # -i image: 入力画像
            # -i audio: 入力音声
            # -c:v libx264 -preset ultrafast: 高速エンコード
            # -tune stillimage: 静止画に最適化
            # -pix_fmt yuv420p: 互換性のため
            # -shortest: 短い方（音声）に合わせる
            
            # ffprobeで音声ファイルの長さを取得
            probe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(audio)
            ]
            result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            audio_duration = float(result.stdout.strip())

            cmd = [
                'ffmpeg',
                '-y',  # 上書き
                '-loop', '1',
                '-i', str(img),
                '-i', str(audio),
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'stillimage',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-t', str(audio_duration), # <-- ここで長さを明示的に指定
                '-shortest',
                '-threads', '0',  # 全CPUコア使用
                str(output_path)
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"動画 {i:03d} を生成しました: {output_path}")
            except subprocess.CalledProcessError as e:
                print(f"エラー: 動画 {i:03d} の生成に失敗しました")
                print(f"stderr: {e.stderr}")
                raise


class FinalVideoGenerator:
    """最終的なプレゼンテーション動画を生成"""
    
    def generate(self, project_name: str):
        """スライド動画を結合して最終動画を生成"""
        gen = PresentationGenerator(project_name)
        
        # 既存の最終動画を削除
        output_path = gen.project_dir / "presentation.mp4"
        if output_path.exists():
            output_path.unlink()
            print(f"既存のpresentation.mp4を削除しました")
        
        # スライド動画を取得
        video_files = sorted(gen.project_dir.glob("video/*.mp4"))
        
        if not video_files:
            print("エラー: スライド動画が見つかりません")
            sys.exit(1)
        
        print(f"{len(video_files)}個の動画を結合中...")
        
        # ffmpegのconcatフィルタ用のファイルリストを作成
        concat_file = gen.project_dir / "concat_list.txt"
        with open(concat_file, 'w', encoding='utf-8') as f:
            for vf in video_files:
                # ffmpegのconcat形式: file 'path'
                f.write(f"file '{vf.absolute()}'\n")
        
        # ffmpegで高速結合（再エンコードなし）
        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',  # 再エンコードなしでコピー（超高速）
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            concat_file.unlink()  # 一時ファイル削除
            print(f"最終動画を生成しました: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"エラー: 動画結合に失敗しました")
            print(f"stderr: {e.stderr}")
            if concat_file.exists():
                concat_file.unlink()
            raise


# コマンドライン実行用関数

def step1_generate_slides(args):
    """ステップ1: スライドMarkdown生成"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("エラー: GEMINI_API_KEY環境変数が設定されていません")
        sys.exit(1)
    
    generator = SlideMarkdownGenerator(api_key)
    project_name = generator.generate(
        Path(args.content_file),
        args.num_slides,
        args.time,
        args.project
    )
    print(f"プロジェクト名: {project_name}")


def step2_generate_narration(args):
    """ステップ2: ナレーションテキスト生成"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("エラー: GEMINI_API_KEY環境変数が設定されていません")
        sys.exit(1)
    
    generator = NarrationTextGenerator(api_key)
    generator.generate(args.project, args.time)


def step3_generate_slide_images(args):
    """ステップ3: スライド画像生成"""
    generator = SlideImageGenerator()
    generator.generate(args.project, args.theme)


def step4_generate_audio(args):
    """ステップ4: ナレーション音声生成"""
    # エンジンパラメータを解析
    engine_kwargs = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine_kwargs['device'] = device

    generator = NarrationAudioGenerator(**engine_kwargs)
    generator.generate(args.project, args.time)


def step5_generate_slide_videos(args):
    """ステップ5: スライド動画生成"""
    generator = SlideVideoGenerator()
    generator.generate(args.project)


def step6_generate_final_video(args):
    """ステップ6: 最終動画生成"""
    generator = FinalVideoGenerator()
    generator.generate(args.project)


def run_all(args):
    """全ステップを実行"""
    print("=== ステップ1: スライドMarkdown生成 ===")
    step1_generate_slides(args)
    
    print("\n=== ステップ2: ナレーションテキスト生成 ===")
    step2_generate_narration(args)
    
    print("\n=== ステップ3: スライド画像生成 ===")
    step3_generate_slide_images(args)
    
    print("\n=== ステップ4: ナレーション音声生成 ===")
    step4_generate_audio(args)
    
    print("\n=== ステップ5: スライド動画生成 ===")
    step5_generate_slide_videos(args)
    
    print("\n=== ステップ6: 最終動画生成 ===")
    step6_generate_final_video(args)
    
    print("\n=== 完了 ===")


def main():
    parser = argparse.ArgumentParser(
        description="プレゼンテーション動画自動生成システム"
    )
    subparsers = parser.add_subparsers(dest="command", help="実行するステップ")
    
    # ステップ1: スライドMarkdown生成
    parser_step1 = subparsers.add_parser("step1", help="スライドMarkdown生成")
    parser_step1.add_argument("content_file", help="プレゼンテーション内容ファイル")
    parser_step1.add_argument("-n", "--num-slides", type=int, help="スライド枚数")
    parser_step1.add_argument("-t", "--time", type=int, default=180, help="想定時間(秒)")
    parser_step1.add_argument("-p", "--project", help="プロジェクト名")
    
    # ステップ2: ナレーションテキスト生成
    parser_step2 = subparsers.add_parser("step2", help="ナレーションテキスト生成")
    parser_step2.add_argument("project", help="プロジェクト名")
    parser_step2.add_argument("-t", "--time", type=int, default=180, help="想定時間(秒)")
    
    # ステップ3: スライド画像生成
    parser_step3 = subparsers.add_parser("step3", help="スライド画像生成")
    parser_step3.add_argument("project", help="プロジェクト名")
    parser_step3.add_argument("--theme", default="default", help="Marpテーマ")
    
    # ステップ4: ナレーション音声生成
    parser_step4 = subparsers.add_parser("step4", help="ナレーション音声生成")
    parser_step4.add_argument("project", help="プロジェクト名")
    parser_step4.add_argument("-t", "--time", type=int, default=180, help="想定時間(秒)")
    
    # ステップ5: スライド動画生成
    parser_step5 = subparsers.add_parser("step5", help="スライド動画生成")
    parser_step5.add_argument("project", help="プロジェクト名")
    
    # ステップ6: 最終動画生成
    parser_step6 = subparsers.add_parser("step6", help="最終動画生成")
    parser_step6.add_argument("project", help="プロジェクト名")
    
    # 全ステップ実行
    parser_all = subparsers.add_parser("all", help="全ステップを実行")
    parser_all.add_argument("content_file", help="プレゼンテーション内容ファイル")
    parser_all.add_argument("-n", "--num-slides", type=int, help="スライド枚数")
    parser_all.add_argument("-t", "--time", type=int, default=180, help="想定時間(秒)")
    parser_all.add_argument("-p", "--project", help="プロジェクト名")
    parser_all.add_argument("--theme", default="default", help="Marpテーマ")
    
    args = parser.parse_args()
    
    if args.command == "step1":
        step1_generate_slides(args)
    elif args.command == "step2":
        step2_generate_narration(args)
    elif args.command == "step3":
        step3_generate_slide_images(args)
    elif args.command == "step4":
        step4_generate_audio(args)
    elif args.command == "step5":
        step5_generate_slide_videos(args)
    elif args.command == "step6":
        step6_generate_final_video(args)
    elif args.command == "all":
        run_all(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
