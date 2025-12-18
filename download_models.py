import os
from pathlib import Path
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from huggingface_hub import hf_hub_download
import pyopenjtalk

print("Downloading Open JTalk dictionary...")
# pyopenjtalkの辞書ダウンロードをトリガー
# g2p関数などを呼ぶと初回のみダウンロードが走るはず
try:
    pyopenjtalk.g2p("こんにちは")
except Exception as e:
    print(f"Open JTalk dictionary download triggered: {e}")

print("Downloading BERT models...")
bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

print("Downloading Style-Bert-Vits2 model assets...")
assets_root = Path("model_assets")
model_file = "jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors"
config_file = "jvnv-F1-jp/config.json"
style_file = "jvnv-F1-jp/style_vectors.npy"

for file in [model_file, config_file, style_file]:
    hf_hub_download("litagin/style_bert_vits2_jvnv", file, local_dir=assets_root)

print("All models downloaded successfully.")
