FROM python:3.10-slim

# システム依存関係のインストール
# ffmpeg: 動画処理用
# nodejs, npm: Marp CLI用
# git: 依存パッケージの取得用
# wget: Chrome取得用
# fonts-noto-cjk: 日本語フォント(スライド生成用)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    nodejs \
    npm \
    git \
    wget \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# Google Chromeのインストール (Marp画像生成用)
RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
    && apt-get update \
    && apt-get install -y ./google-chrome-stable_current_amd64.deb \
    && rm google-chrome-stable_current_amd64.deb \
    && rm -rf /var/lib/apt/lists/*

# Marp CLIのインストール
RUN npm install -g @marp-team/marp-cli

# 作業ディレクトリの設定
WORKDIR /app

# Python依存関係のインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# モデルの事前ダウンロード（キャッシュ）
COPY download_models.py .
RUN python download_models.py && rm download_models.py

# アプリケーションコードのコピー
COPY . .

# 実行権限の付与
RUN chmod +x run.sh

# デフォルトコマンド
CMD ["/bin/bash"]
