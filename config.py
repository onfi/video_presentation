"""
プレゼンテーション動画自動生成システムの設定ファイル
"""

# 定数: 日本語の読み上げ速度（文字/分）
JAPANESE_READING_SPEED = 400

# Marp Front Matter テンプレート
MARP_FRONT_MATTER = """---
marp: true
theme: default
lang: ja
style: |
  h1, h2, h3, h4, h5, h6, p, li {{
    word-break: auto-phrase;
  }}
  section {{
    font-size: 36px;
  }}
---"""

# プロンプトテンプレート
class PromptTemplates:
    """各種プロンプトのテンプレート"""
    
    @staticmethod
    def slide_generation(content: str, num_slides: int, chars_per_slide: int, time_per_slide: float) -> str:
        """スライド生成用プロンプト"""
        return f"""以下のプレゼンテーション内容に基づいて、Marp形式のMarkdownでスライドを作成してください。

プレゼンテーション内容:
{content}

要件:
- {num_slides}枚のスライドを作成
- Marp形式（各スライドは`---`で区切る）
- 日本語で作成
- タイトルスライドを含める。著者名や日付は不要
- 各スライドは簡潔で視覚的に分かりやすく
- 箇条書きや図表を適切に使用
- スライドMarkdownのみを出力する
"""
    
    @staticmethod
    def slide_schema(num_slides: int) -> dict:
        """スライド生成用のレスポンススキーマ"""
        return {
            "type": "object",
            "properties": {
                "markdown": {
                    "type": "string",
                    "description": f"""Marp形式のMarkdown。{num_slides}枚のスライドを`---`で区切る。
Front Matterは以下の形式で必ず含める:
{MARP_FRONT_MATTER}
"""
                }
            },
            "required": ["markdown"]
        }
    
    @staticmethod
    def narration_generation(slides_text: str, num_slides: int, chars_per_slide: int, time_per_slide: float) -> str:
        """ナレーション生成用プロンプト"""
        return f"""以下は{num_slides}枚のプレゼンテーションスライドです。
全スライド（1～{num_slides}）のナレーション原稿を作成してください。

【絶対厳守ルール】
1. 挨拶はスライド1の冒頭のみ。スライド2以降は禁止：
   ❌ 禁止表現: 「本日は」「ご紹介」「ご説明」「まず」「はじめに」「次に」「それでは」「について」「見ていきましょう」
   ⭕ スライド2以降: 前のスライドから直接続く内容のみ
   
2. 文字数制限を厳守:
   - 最小{chars_per_slide-20}文字、最大{chars_per_slide+20}文字（目標{chars_per_slide}文字）
   - この範囲を超えないこと
   
3. 内容の重複禁止:
   - 各スライドは固有の情報のみ
   - 前のスライドで述べたことは繰り返さない

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
                    "description": f"全{num_slides}個のナレーション。文字数: 最小{chars_per_slide-20}、最大{chars_per_slide+20}を厳守。スライド1のみ挨拶、2以降は挨拶・導入表現を一切使わず前のスライドから自然に続ける。内容重複禁止。",
                    "items": {
                        "type": "object",
                        "properties": {
                            "slide": {
                                "type": "integer",
                                "description": f"スライド番号（1～{num_slides}）"
                            },
                            "text": {
                                "type": "string",
                                "description": f"ナレーション原稿。文字数: 必ず{chars_per_slide-20}～{chars_per_slide+20}文字の範囲内。スライド1のみ挨拶可、2以降は挨拶・導入禁止で前スライドの続きとして開始。アルファベットは使用不可、カタカナで表記すること。",
                                "minLength": chars_per_slide - 20,
                                "maxLength": chars_per_slide + 20
                            }
                        },
                        "required": ["slide", "text"]
                    }
                }
            },
            "required": ["narrations"]
        }
