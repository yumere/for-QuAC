python run_coqa.py \
  --model_type bert \
  --model_name_or_path C://Users/shwks/Desktop/github/ConversationalQA/BERT_CoQA/pytorch-transformers-master/examples/coqa_output/checkpoint-24700/ \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file C://Users/shwks/Desktop/Dataset/CoQA/coqa_train.json \
  --predict_file C://Users/shwks/Desktop/Dataset/CoQA/coqa_dev.json \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./coqa_output_additional_train \
  --per_gpu_eval_batch_size 8 \
  --per_gpu_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --overwrite_output_dir \