cd /aifs4su/hansirui_4th/dumengfei/code/starVLA

source /aifs4su/hansirui_4th/miniconda3/bin/activate
conda activate starVLA

NUM_PROCESSES=8 \
DATA_MIX=libero_90 \
RUN_ID=0615_qwenfast_libero90_ep10 \
MAX_TRAIN_STEPS=46110 \
SAVE_INTERVAL=4611 \
BATCH_SIZE=16 \
WANDB_MODE=offline \
MAIN_PROCESS_PORT=33376 \
bash examples/LIBERO/train_files/run_libero_qwenfast_train.sh