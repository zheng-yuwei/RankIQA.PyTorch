# 用单GPU进行模型评估（没GPU则自动退化为CPU模型评估）
python main.py --data data -e --arch mobilenetv3_large --num_classes 1 \
               --criterion=rank --margin 0.1 \
               --image_size 224 224 --batch_size 256 --workers 16 \
               --resume checkpoints/checkpoint_mobilenetv3_large.pth \
               -g 1 -n 1
