export HF_ENDPOINT=https://hf-mirror.com

mkdir "./ckpts"
huggingface-cli download Tencent-Hunyuan/HunyuanDiT --local-dir ./ckpts
