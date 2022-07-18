export TOKENIZERS_PARALLELISM=true
export PYTORCH_ENABLE_MPS_FALLBACK=1
python -m src.main --overwrite 1 --debug 1 --max_seq_length 256 --device cuda