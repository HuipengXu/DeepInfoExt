export TOKENIZERS_PARALLELISM=true
export PYTHONPATH="../deep_info_ext"
export OMP_NUM_THREADS=1
# python3 -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 -m src.main \
#         --overwrite 0 \
#         --debug 0 \
#         --max_seq_length 128 \
#         --device 0,1 \
#         --do_train 1 \
#         --model_path ptms/roberta-base \
#         --num_workers 4 \
#         --learning_rate 2e-5 \
#         --batch_size 32 \
#         --num_train_epochs 5 \
#         --logging_steps 500
# torchrun --nnodes=1 --nproc_per_node=2 -m deep_info_ext.ner.bert_softmax.main \
#         --overwrite 0 \
#         --debug 0 \
#         --max_seq_length 128 \
#         --device cpu \
#         --do_train 1 \
#         --model_path ptms/roberta-base \
#         --num_workers 4 \
#         --learning_rate 2e-5 \
#         --batch_size 32 \
#         --num_train_epochs 5 \
#         --logging_steps 500 \
#         --train_file msra_train_bio.txt \
#         --test_file msra_test_bio.txt \
#         --num_train_examples 45000 \
#         --num_test_examples 3442
python3 -m deep_info_ext.ner.bert_softmax.main \
        --overwrite 1 \
        --debug 1 \
        --max_seq_length 128 \
        --device cpu \
        --do_train 1 \
        --model_path ptms/rbt3 \
        --num_workers 4 \
        --learning_rate 2e-5 \
        --batch_size 32 \
        --num_train_epochs 1 \
        --logging_steps 5 \
        --train_file msra_train_bio.txt \
        --test_file msra_test_bio.txt \
        --num_train_examples 45000 \
        --num_test_examples 3442