export TOKENIZERS_PARALLELISM=true
python3 -m deep_info_ext.ner.bert_crf.main \
        --overwrite 1 \
        --debug 0 \
        --max_seq_length 512 \
        --device 0 \
        --do_train 1 \
        --model_path ptms/roberta-base \
        --num_workers 4 \
        --learning_rate 2e-5 \
        --max_grad_norm 1.0 \
        --batch_size 16 \
        --num_train_epochs 5 \
        --logging_steps 500 \
        --train_file msra_train_bio.txt \
        --test_file msra_test_bio.txt \
        --num_train_examples 45000 \
        --num_test_examples 3442