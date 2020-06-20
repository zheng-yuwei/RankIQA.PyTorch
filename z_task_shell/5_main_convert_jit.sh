# 将指定模型文件转为JIT格式
python main.py --jit --arch mobilenetv3_large --num_classes 1 \
               --image_size 224 224 \
               --resume checkpoints/model_best_mobilenetv3_large.pth \
               -g 0