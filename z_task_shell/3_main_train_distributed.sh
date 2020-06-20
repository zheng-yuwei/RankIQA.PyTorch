# 分布式模型训练
CUDA_VISIBLE_DEVICES=0,1 python main.py --data data/ --train \
    --arch mobilenetv3_large --num_classes 1 \
    --criterion=rank --margin 0.1 \
    --image_size 224 224 \
    --pretrained \
    --warmup 5 --epochs 65 -b 72 -j 10 \
    --gpus 2 --nodes 1 \
    --init_method  tcp://11.6.118.36:10006  # tcp://11.6.127.208:10006 \
    --sync_bn \
    --rank 0
