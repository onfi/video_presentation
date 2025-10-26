#!/bin/bash

# プレゼンテーション動画自動生成 実行スクリプト

set -e  # エラーで停止

# 色付き出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================${NC}"
echo -e "${BLUE}プレゼンテーション動画自動生成${NC}"
echo -e "${BLUE}==================================${NC}\n"

# 環境変数チェック
# .env があれば読み込む
if [ -f .env ]; then
    # shellcheck disable=SC1091
    set -o allexport
    source .env
    set +o allexport
fi

# 環境変数チェック
if [ -z "$GEMINI_API_KEY" ]; then
    echo -e "${RED}エラー: GEMINI_API_KEY環境変数が設定されていません${NC}"
    echo -e "以下のコマンドで設定してください:"
    echo -e "  export GEMINI_API_KEY='your-api-key-here'"
    exit 1
fi

# 引数チェック
if [ $# -lt 1 ]; then
    echo -e "${YELLOW}使い方:${NC}"
    echo -e "  $0 <content_file> [options]"
    echo ""
    echo -e "${YELLOW}オプション:${NC}"
    echo -e "  -p, --project <name>      プロジェクト名"
    echo -e "  -n, --num-slides <num>    スライド枚数"
    echo -e "  -t, --time <seconds>      動画時間（秒）"
    echo -e "  --theme <theme>           Marpテーマ"
    echo -e "  --step <1-6>              特定のステップのみ実行"
    echo ""
    echo -e "${YELLOW}例:${NC}"
    echo -e "  $0 example_presentation.txt"
    echo -e "  $0 example_presentation.txt -p my_project -t 300"
    echo -e "  $0 example_presentation.txt --step 4"
    exit 1
fi

CONTENT_FILE=$1
shift

# デフォルト値
PROJECT=""
NUM_SLIDES=""
TIME="180"
THEME="default"
STEP=""

# 引数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--project)
            PROJECT="$2"
            shift 2
            ;;
        -n|--num-slides)
            NUM_SLIDES="$2"
            shift 2
            ;;
        -t|--time)
            TIME="$2"
            shift 2
            ;;
        --theme)
            THEME="$2"
            shift 2
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}エラー: 未知のオプション: $1${NC}"
            exit 1
            ;;
    esac
done

# プロジェクト名の決定
if [ -z "$PROJECT" ]; then
    PROJECT=$(basename "$CONTENT_FILE" | sed 's/\.[^.]*$//')
fi

# コマンド構築
PYTHON="python"
# 仮想環境があればそれを優先
if [ -x "./venv/bin/python" ]; then
    PYTHON="./venv/bin/python"
fi

CMD="$PYTHON main.py"


# 完了メッセージ関数
print_completion_message() {
    echo -e "\n${GREEN}==================================${NC}"
    echo -e "${GREEN}処理が完了しました！${NC}"
    echo -e "${GREEN}==================================${NC}"
    echo -e "出力ディレクトリ: ${BLUE}./outputs/$PROJECT/${NC}"
    echo -e "最終動画: ${BLUE}./outputs/$PROJECT/presentation.mp4${NC}"
}

# 特定のステップのみ実行
if [ -n "$STEP" ]; then
    echo -e "${GREEN}ステップ $STEP を実行します${NC}\n"
    
    case $STEP in
        1)
            ARGS="$CONTENT_FILE"
            [ -n "$NUM_SLIDES" ] && ARGS="$ARGS -n $NUM_SLIDES"
            ARGS="$ARGS -t $TIME -p $PROJECT"
            $CMD step1 $ARGS
            ;;
        2)
            $CMD step2 $PROJECT -t $TIME
            ;;
        3)
            $CMD step3 $PROJECT --theme $THEME
            ;;
        4)
            ARGS="$PROJECT -t $TIME"
            $CMD step4 $ARGS
            ;;
        5)
            $CMD step5 $PROJECT
            ;;
        6)
            $CMD step6 $PROJECT
            print_completion_message
            ;;
        *)
            echo -e "${RED}エラー: ステップは1-6の範囲で指定してください${NC}"
            exit 1
            ;;
    esac
else
    # 全ステップ実行
    echo -e "${GREEN}全ステップを実行します${NC}\n"
    
    ARGS="$CONTENT_FILE"
    [ -n "$NUM_SLIDES" ] && ARGS="$ARGS -n $NUM_SLIDES"
    ARGS="$ARGS -t $TIME"
    [ -n "$PROJECT" ] && ARGS="$ARGS -p $PROJECT"
    ARGS="$ARGS --theme $THEME"
    
    $CMD all $ARGS
    print_completion_message
fi
