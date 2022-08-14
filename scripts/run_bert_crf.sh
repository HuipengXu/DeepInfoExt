export TOKENIZERS_PARALLELISM=true
python3 -m deep_info_ext.ner.bert_crf.main \
        --overwrite 1 \
        --debug 1 \
        --max_seq_length 128 \
        --device cpu \
        --do_train 0 \
        --model_path ptms/rbt3 \
        --num_workers 1 \
        --learning_rate 2e-5 \
        --crf_lr 0.1 \
        --batch_size 8 \
        --num_train_epochs 1 \
        --logging_steps 5 \
        --train_file msra_train_bio.txt \
        --test_file msra_test_bio.txt \
        --num_train_examples 45000 \
        --num_test_examples 3442