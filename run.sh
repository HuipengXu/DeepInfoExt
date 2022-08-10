export TOKENIZERS_PARALLELISM=true
# export OMP_NUM_THREADS=1
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
torchrun --nnodes=1 --nproc_per_node=2 -m src.main \
        --overwrite 0 \
        --debug 0 \
        --max_seq_length 128 \
        --device 0,1 \
        --do_train 1 \
        --model_path ptms/roberta-base \
        --num_workers 4 \
        --learning_rate 2e-5 \
        --batch_size 32 \
        --num_train_epochs 5 \
        --logging_steps 500
# python3 -m src.main \
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