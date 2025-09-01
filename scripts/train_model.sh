GPU_IDS=-1

DATAROOT=../GAPs384
NAME=gaps384_40x40_deepcrack
MODEL=deepcrack
DATASET_MODE=gaps384

BATCH_SIZE=1
NUM_CLASSES=1
LOAD_WIDTH=400
LOAD_HEIGHT=400
LOSS_MODE=focal

NORM=batch
NITER=30
NITER_DECAY=10

python3 train.py \
  --dataroot ${DATAROOT} \
  --name ${NAME} \
  --model ${MODEL} \
  --dataset_mode ${DATASET_MODE} \
  --gpu_ids ${GPU_IDS} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --batch_size ${BATCH_SIZE} \
  --num_classes ${NUM_CLASSES} \
  --norm ${NORM} \
  --lr_decay_iters 40 \
  --lr_policy step \
  --load_width ${LOAD_WIDTH} \
  --load_height ${LOAD_HEIGHT} \
  --no_flip 1 \
  --display_id 0 \
  --loss_mode ${LOSS_MODE} 