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

-----

Shan will run on Feb 28 morning:

Train from beginning (no checkpoint), with 2 epochs (NO evaluate during training flag). Be sure tmp/debug_squad5 does not exist, otherwise use debug_squad6:

python run_squad.py --model_type bert --model_name_or_path bert-base-uncased --do_lower_case --version_2_with_negative --do_train --do_eval --train_file /home/cs224nproject/cs224n_finalproject/transformers/examples/tests_samples/SQUAD/train-v2.0big.json --predict_file /home/cs224nproject/cs224n_finalproject/transformers/examples/tests_samples/SQUAD/dev-v2.0big.json --per_gpu_train_batch_size 1 --learning_rate 4e-5 --num_train_epochs 2.0 --overwrite_output_dir --max_seq_length 384 --doc_stride 128 --output_dir /home/cs224nproject/cs224n_finalproject/transformers/examples/tmp/debug_squad5 --save_steps 5000 
