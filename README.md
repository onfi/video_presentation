# プレゼンテーション動画自動生成システム

Text to SpeechとMarpを使用して、日本語プレゼンテーション動画を自動生成するシステムです。

## 機能

- Gemini APIを使用したスライドとナレーション原稿の自動生成
- Marpによる美しいスライド画像の生成
- Hugging FaceのTTSモデルによる音声合成
- 自動的な動画編集と結合

## セットアップ

### 1. 必要なソフトウェア

```bash
# Python 3.8以上
python --version

# Node.js (Marp CLI用)
node --version
npm --version
```

### 2. Pythonパッケージのインストール

```bash
pip install -r requirements.txt
```

### 3. Marp CLIのインストール

```bash
npm install -g @marp-team/marp-cli
```

### 4. rubberband-cliのインストール (pyrubberband用)

pyrubberbandは高品質なタイムストレッチ/ピッチシフトを提供しますが、内部で`rubberband-cli`に依存します。OSに応じてインストールしてください。

- **Ubuntu/Debian:**
  ```bash
  sudo apt-get install rubberband-cli
  ```
- **macOS (Homebrew):**
  ```bash
  brew install rubberband
  ```
- **Windows:**
  公式サイトからダウンロードするか、`scoop install rubberband` (Scoopがインストールされている場合)

### 5. 環境変数の設定

Gemini APIキーを環境変数に設定します：

```bash
export GEMINI_API_KEY="your-api-key-here"
```

または`.env`ファイルを作成：

```
GEMINI_API_KEY=your-api-key-here
```

## 使い方

### 基本的な使い方

プレゼンテーション内容を記載したテキストファイルを用意し、以下のコマンドで全自動生成：

```bash
python main.py all presentation_content.txt
```

### ステップごとの実行

各ステップを個別に実行することも可能です：

#### ステップ1: スライドMarkdown生成

```bash
python main.py step1 presentation_content.txt -n 10 -t 300 -p my_presentation
```

オプション：
- `-n, --num-slides`: スライド枚数（省略時はAI判断）
- `-t, --time`: 想定動画時間（秒、デフォルト180秒）
- `-p, --project`: プロジェクト名（省略時はファイル名）

#### ステップ2: ナレーション原稿生成

```bash
python main.py step2 my_presentation
```

#### ステップ3: スライド画像生成

```bash
python main.py step3 my_presentation --theme default
```

オプション：
- `--theme`: Marpテーマ（default, gaia, uncover等）
- `--style`: カスタムCSSファイルを指定（例: `--style custom.css`）。Markdownファイル内の`style`ブロックもテーマを上書き/拡張します。

#### ステップ4: ナレーション音声生成

```bash
python main.py step4 my_presentation --model microsoft/speecht5_tts -t 300
```

オプション：
- `--model`: TTSモデル名
- `-t, --time`: 想定動画時間（秒）

#### ステップ5: スライド動画生成

```bash
python main.py step5 my_presentation
```

#### ステップ6: 最終動画生成

```bash
python main.py step6 my_presentation
```

## プロジェクト構造

```
./outputs/
└── {project_name}/
    ├── slides.md              # 生成されたMarkdown
    ├── narration000.txt       # ナレーション原稿
    ├── narration001.txt
    ├── ...
    ├── slides/                # スライド画像
    │   ├── 000.png
    │   ├── 001.png
    │   └── ...
    ├── audio/                 # 音声ファイル
    │   ├── slide000.mp3
    │   ├── slide001.mp3
    │   └── ...
    ├── video/                 # スライドごとの動画
    │   ├── slide000.mp4
    │   ├── slide001.mp4
    │   └── ...
    └── presentation.mp4       # 最終動画
```

## TTSエンジンの選択

このシステムは複数のTTSエンジンをサポートしています。推奨順：

### 1. Style-Bert-Vits2 (推奨、高品質な日本語TTS)

`pyrubberband`を使用することで、より高品質な速度調整が可能です。

```bash
# 使用
python main.py step4 my_presentation --engine style-bert-vits2
```

### 2. VOICEVOX (高品質な日本語音声合成)

**最も高品質な日本語音声合成**

```bash
# VOICEVOXをダウンロード・インストール
# https://voicevox.hiroshiba.jp/

# VOICEVOXを起動後
python main.py step4 my_presentation --engine voicevox --speaker-id 1
```

話者ID例：
- 1: 四国めたん（ノーマル）
- 3: ずんだもん（ノーマル）
- 8: 春日部つむぎ（ノーマル）

### 3. ESPnet (高品質な日本語TTS)

```bash
# インストール
pip install espnet espnet_model_zoo

# 使用
python main.py step4 my_presentation --engine espnet
```

### 4. gTTS (簡単、要インターネット)

```bash
# インストール
pip install gTTS pydub

# 使用
python main.py step4 my_presentation --engine gtts
```

### 5. SpeechT5 (多言語対応)

```bash
# 使用
python main.py step4 my_presentation --engine speecht5
```

### 自動選択

```bash
# 利用可能なエンジンを自動で選択
python main.py step4 my_presentation --engine auto
```il_hop_length256_sampling_rate24000`（日本語特化）

## トラブルシューティング

### Marp CLIが見つからない

```bash
npm install -g @marp-team/marp-cli
```

### CUDA out of memory エラー

GPUメモリが不足している場合、CPUで実行されます。より軽量なTTSモデルを選択してください。

### 音声が生成されない

各TTSエンジンが正しくインストール・起動されているか確認してください：

- **VOICEVOX**: アプリケーションが起動していますか？
- **ESPnet**: `pip install espnet espnet_model_zoo` でインストール済みですか？
- **gTTS**: インターネット接続はありますか？

### VOICEVOX接続エラー

VOICEVOXアプリケーションが起動していることを確認してください。デフォルトではポート50021で待ち受けています。

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します！
