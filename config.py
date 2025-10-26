"""
プレゼンテーション動画自動生成システムの設定ファイル
"""
from typing import Optional

# 定数: 日本語の読み上げ速度（文字/分）
JAPANESE_READING_SPEED = 450

# プロンプトテンプレート
class PromptTemplates:
    """各種プロンプトのテンプレート"""
    
    @staticmethod
    def slide_generation(content: str, num_slides: Optional[int], chars_per_slide: int, time_per_slide: float) -> str:
        """スライド生成用プロンプト"""
        
        requirements = []
        if num_slides is not None:
            requirements.append(f"- {num_slides}枚のスライドを作成")
        
        requirements.extend([
            "- Marp形式、Marpで出力したとき魅力的なスライドになるようにコーディングする",
            "- 日本語で作成",
            "- タイトルスライドを含める。著者名や日付は不要",
            "- 各スライドは視覚的に分かりやすく、箇条書きや図表を適切に使用",
        ])
        
        return f"""以下のプレゼンテーション内容に基づいて、Marp形式のMarkdownでスライドを作成してください。

プレゼンテーション内容:
{content}

要件:
{chr(10).join(requirements)}
"""
    
    @staticmethod
    def slide_schema(num_slides: int) -> dict:
        """スライド生成用のレスポンススキーマ"""
        return {
            "type": "object",
            "properties": {
                "markdown": {
                    "type": "string",
                    "description": f"""プレゼンテーションスライド。{num_slides}枚のスライド。Marp形式のMarkdown"""
                }
            },
            "required": ["markdown"]
        }
    
    @staticmethod
    def narration_generation(slides_text: str, num_slides: int, chars_per_slide: int, time_per_slide: float) -> str:
        """ナレーション生成用プロンプト"""
        return f"""以下は{num_slides}枚のプレゼンテーションスライドです。
全スライド（1～{num_slides}）のナレーション原稿を作成してください。

1. 挨拶はスライド1の冒頭のみ。スライド2以降は禁止
2. 文字数制限を厳守: 1スライドあたり{chars_per_slide}文字前後
3. 各スライドの前後の流れを考え、連続して同じ説明をしない

プレゼンテーション全体のスライド:
{slides_text}
"""
    
    @staticmethod
    def narration_schema(num_slides: int, chars_per_slide: int, time_per_slide: float) -> dict:
        """ナレーション生成用のレスポンススキーマ"""
        return {
            "type": "object",
            "properties": {
                "narrations": {
                    "type": "array",
                    "description": f"全{num_slides}個のナレーション。文字数: 最小{chars_per_slide-20}、最大{chars_per_slide+20}を厳守。スライド1のみ挨拶、2以降は挨拶・導入表現を一切使わず前のスライドから自然に続ける。内容重複禁止。アルファベットからの表記は禁止、英語はカタカナで表記。1文ごとに改行する。",
                    "items": {
                        "type": "object",
                        "properties": {
                            "slide": {
                                "type": "integer",
                                "description": f"スライド番号（1～{num_slides}）"
                            },
                            "text": {
                                "type": "string",
                                "description": f"ナレーション原稿。文字数: 必ず{chars_per_slide}文字前後にする。スライド1のみ挨拶可、2以降は挨拶・導入禁止で前スライドの続きとして開始。アルファベットは使用不可、英語・コマンド・アルファベットの製品名はカタカナで表記する。",
                                "minLength": int(chars_per_slide * 0.9),
                                "maxLength": int(chars_per_slide * 1.1)
                            }
                        },
                        "required": ["slide", "text"]
                    }
                }
            },
            "required": ["narrations"]
        }
