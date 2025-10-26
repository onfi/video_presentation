# プレゼンテーション動画自動生成システム

このシステムは、Text-to-Speech (TTS) 技術とMarpを組み合わせることで、日本語のプレゼンテーション動画を自動で生成します。企画書や発表原稿から、視覚的に魅力的なスライドと自然なナレーションを含む動画コンテンツを効率的に作成し、コンテンツ制作の負担を大幅に軽減します。

## 主な機能

- Gemini APIを使用したスライドとナレーション原稿の自動生成
- Marpによる美しいスライド画像の生成
- Hugging FaceのTTSモデルによる音声合成
- 自動的な動画編集と結合

## セットアップ

本システムを利用するためのセットアップ手順を以下に示します。

### 1. 必要なソフトウェアのインストール

以下のソフトウェアがシステムにインストールされていることを確認してください。

-   **Python 3.9以上**
    ```bash
    python --version
    ```
-   **Node.js (Marp CLI用)**
    ```bash
    node --version
    npm --version
    ```

### 2. Pythonパッケージのインストール

プロジェクトのルートディレクトリで、以下のコマンドを実行し、必要なPythonパッケージをインストールします。

```bash
pip install -r requirements.txt
```

### 3. Marp CLIのインストール

Marp CLIはスライド画像を生成するために必要です。以下のコマンドでグローバルにインストールします。

```bash
npm install -g @marp-team/marp-cli
```

### 4. 環境変数の設定

Gemini APIキーを設定する必要があります。以下のいずれかの方法で設定してください。

#### 方法A: 環境変数として設定

```bash
export GEMINI_API_KEY="your-api-key-here"
```

#### 方法B: `.env`ファイルを作成

プロジェクトのルートディレクトリに`.env`という名前のファイルを作成し、以下の内容を記述します。

```
GEMINI_API_KEY=your-api-key-here
```

**注意:** `your-api-key-here` を実際のGemini APIキーに置き換えてください。`.env`ファイルはGit管理から除外することをお勧めします（`.gitignore`に`*.env`を追加するなど）。

## 使い方

本システムは、プレゼンテーションの内容を記述したテキストファイル（例: `presentation_content.txt`）を入力として受け取ります。このファイルには、スライドの各ページに対応する内容と、それぞれのナレーション原稿を含めることができます。

### 基本的な使い方

以下のコマンドを実行すると、入力ファイルに基づいてスライドの生成から最終的な動画の結合まで、全てのステップを自動で実行します。

```bash
python main.py all presentation_content.txt
```

### ステップごとの実行

各処理ステップを個別に実行することも可能です。これにより、特定の段階での調整やデバッグが容易になります。

#### ステップ1: スライドMarkdown生成

Gemini APIを使用して、入力テキストファイルからMarp形式のMarkdownスライドとナレーション原稿を生成します。

```bash
python main.py step1 presentation_content.txt -n 10 -t 300 -p my_presentation
```

**オプション:**
-   `-n, --num-slides`: 生成するスライドの枚数を指定します。省略した場合、AIが内容に基づいて最適な枚数を判断します。
-   `-t, --time`: 想定する動画の総時間（秒）を指定します。デフォルトは180秒です。
-   `-p, --project`: プロジェクト名を指定します。この名前で`outputs/`ディレクトリ内に専用のフォルダが作成されます。省略した場合、入力ファイル名がプロジェクト名として使用されます。

#### ステップ2: ナレーション原稿生成

生成されたスライドMarkdownから、各スライドに対応するナレーション原稿を抽出・整理します。

```bash
python main.py step2 my_presentation
```

#### ステップ3: スライド画像生成

Marp CLIを使用して、生成されたMarkdownスライドからPNG形式の画像ファイルを生成します。

```bash
python main.py step3 my_presentation --theme default
```

**オプション:**
-   `--theme`: Marpのテーマを指定します（例: `default`, `gaia`, `uncover`など）。
-   `--style`: カスタムCSSファイルを指定して、スライドのスタイルを詳細に調整できます（例: `--style custom.css`）。Markdownファイル内の`style`ブロックもテーマを上書きまたは拡張するために使用できます。

#### ステップ4: ナレーション音声生成

各スライドのナレーション原稿に基づいて、選択されたTTSエンジンで音声ファイルを生成します。

```bash
python main.py step4 my_presentation --model microsoft/speecht5_tts -t 300
```

**オプション:**
-   `--model`: 使用するTTSモデルの名前を指定します。
-   `-t, --time`: 想定する動画の総時間（秒）を指定します。この時間に合わせて音声の速度が調整されます。

#### ステップ5: スライド動画生成

生成されたスライド画像とナレーション音声ファイルを結合し、各スライドに対応する短い動画ファイルを生成します。

```bash
python main.py step5 my_presentation
```

#### ステップ6: 最終動画生成

全ての個別のスライド動画を結合し、最終的なプレゼンテーション動画ファイルを作成します。

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
