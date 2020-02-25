# cs224n_finalproject



python run_squad.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file tests_samples/SQUAD/dev-v2.0.json \
  --predict_file tests_samples/SQUAD/train-v2.0.json \
  --per_gpu_train_batch_size 1 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/

Forked from: https://github.com/huggingface/transformers
