# 用环境变量指定GPU，进行单GPU训练（没有GPU则退化为CPU训练）
CUDA_VISIBLE_DEVICES=1 python main.py --data data/ --train \
    --arch mobilenetv3_large --num_classes 1 \
    --criterion=rank --margin 0.1 \
    --image_size 224 224 \
    --pretrained \
    --warmup 5 --epochs 65 -b 5 -j 0 \
    --gpus 1 --nodes 1
