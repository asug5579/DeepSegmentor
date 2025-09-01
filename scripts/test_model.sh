GPU_IDS=-1

DATAROOT=../GAPs384
NAME=gaps384_40x40_deepcrack
MODEL=deepcrack
DATASET_MODE=gaps384
PHASE=test

NORM=batch
BATCH_SIZE=1
NUM_CLASSES=1
LOAD_WIDTH=400
LOAD_HEIGHT=400

EPOCH=latest

python3 test.py \
  --dataroot ${DATAROOT} \
  --name ${NAME} \
  --model ${MODEL} \
  --dataset_mode ${DATASET_MODE} \
  --gpu_ids ${GPU_IDS} \
  --batch_size ${BATCH_SIZE} \
  --num_classes ${NUM_CLASSES} \
  --load_width ${LOAD_WIDTH} \
  --load_height ${LOAD_HEIGHT} \
  --phase ${PHASE} \
  --epoch ${EPOCH} \
  --norm ${NORM} 