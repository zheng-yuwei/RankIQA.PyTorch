# 基于PyTorch实现的RankIQA

功能说明：

- 对图像质量进行对比排序（Rank）训练；
- 若图像质量有对应的单个质量评分，可对质量分进行回归预测（regress）训练；
- 若图像质量有多个质量评分，构成一个直方图分布，可对分布进行EMD拟合（emd）训练；
- 训练完毕的网络，可借助`application/convert.py`应用，将模型转为JIT格式；
- 最后，使用`demos/image_assessment.py`进行部署调用；


可用的骨干网络包括：

- [x] PyTorch自带的网络：resnet, shufflenet, densenet, mobilenet, mnasnet等；
- [x] MobileNet v3；
- [x] EfficientNet系列；
- [x] ResNeSt系列;

---

## 包含特性

- 图像的对比排序训练，损失函数为`RankingLoss`，调用了PyTorch的`nn.margin_rank_loss`；
- 图像预测评分和目标评分分布的推土距离损失，损失函数为`EMDLoss`，
  损失函数参考了[NIMA: Neural Image Assessment](https://arxiv.org/abs/1709.05424)的目标函数；
- 图片单个评分的回归损失，这个支持两种预测结果（num_classes=1直接裸输出作为预测分 和 num_classes>1求softmax概率加权平均），
  与单个评分的回归，用mse做损失函数。

---

## 文件结构说明

- `applications`: 包括`test.py, train.py, convert.py`等应用，提供给`main.py`调用；
- `checkpoints`: 训练好的模型文件保存目录（当前可能不存在）；
- `criterions`: 自定义损失函数，目前主要包含 `ranking_loss`, `emd_loss`, `regress_loss`；
- `data`: 训练/测试/验证/预测等数据集存放的路径，数据集格式查看下面的`使用说明/数据准备`；
- `dataloader`: 自定义数据集`Dataset`子类和数据集加载`DataLoader`子类，保证在不同任务中加载不同格式的数据文件进行训练；
- `demos`: 模型使用的demo，目前`image_assessment.py`显示如何调用`jit`格式模型进行预测；
- `logs`: 训练过程中TensorBoard日志存放的文件（当前可能不存在）；
- `models`: 自定义的模型结构；
- `optim`: 一些前沿的优化器，PyTorch官方还未实现，RAdam可尝试；
- `pretrained`: 预训练模型文件（当前可能不存在）；
- `utils`: 工具脚本：图片数据校验`check_images.py`、日志`my_logger.py`、数据生成`data_generate`等；
- `config.py`: 配置文件；
- `main.py`: 总入口；
- `requirements.txt`: 工程依赖包列表；

---

## 使用说明

### 数据准备

在文件夹`data`下放数据，分成三个文件夹: `train/test/val`，对应 训练/测试/验证 数据文件夹；
每个子文件夹下，需要根据训练任务的不同放置训练图像（若需要修改数据读取方式等，可查看`data/my_dataloader`）。

数据准备完毕后，使用`utils/check_images.py`脚本，检查图像数据的有效性，防止在训练过程中遇到无效图片中止训练。

#### 排序对比损失任务

在每个数据集文件夹根目录下放置标签文件：如训练集`data/train/label.txt`，每行两张图像的项目相对路径，其中第一个图像质量>第二个图像质量，
内容举例如下：

```text
data/train/refimgs/carnivaldolls.bmp,data/train/gaussian_noise/gaussian_noise5/carnivaldolls.bmp
data/train/refimgs/carnivaldolls.bmp,data/train/gaussian_noise/gaussian_noise7/carnivaldolls.bmp
data/train/refimgs/carnivaldolls.bmp,data/train/gaussian_noise/gaussian_noise11/carnivaldolls.bmp
data/train/refimgs/carnivaldolls.bmp,data/train/gaussian_noise/gaussian_noise15/carnivaldolls.bmp
data/train/refimgs/carnivaldolls.bmp,data/train/gaussian_noise/gaussian_noise21/carnivaldolls.bmp
data/train/refimgs/carnivaldolls.bmp,data/train/white_noise/white_noise0/carnivaldolls.bmp
data/train/refimgs/carnivaldolls.bmp,data/train/white_noise/white_noise3/carnivaldolls.bmp
data/train/refimgs/carnivaldolls.bmp,data/train/white_noise/white_noise5/carnivaldolls.bmp
data/train/refimgs/carnivaldolls.bmp,data/train/white_noise/white_noise7/carnivaldolls.bmp
data/train/refimgs/carnivaldolls.bmp,data/train/white_noise/white_noise9/carnivaldolls.bmp
...
```

#### 分布的推土距离损失任务

在每个数据集文件夹根目录下放置标签文件：如训练集`data/train/label.txt`，每行：一张图像的项目相对路径，这张图的MOS概率分布，
内容举例如下：

```text
data/train/refimgs/carnivaldolls.bmp,0.,0.,0.,0.2,0.8
data/train/gaussian_noise/gaussian_noise7/carnivaldolls.bmp,0.,0.,0.,0.2,0.8
...
```
表示 图像`data/train/refimgs/carnivaldolls.bmp`的评分[1，2，3，4，5]对应的分布为[0.,0.,0.,0.2,0.8]。
注意评分是`1:N+1,N=num_classes`，这与我损失函数`criterions/emd_loss.py`的编写相关。

#### 回归损失任务

在每个数据集文件夹根目录下放置标签文件：如训练集`data/train/label.txt`，每行：一张图像的项目相对路径，这张图的MOS分值
内容举例如下：

```text
data/train/refimgs/carnivaldolls.bmp,7.6
data/train/gaussian_noise/gaussian_noise7/carnivaldolls.bmp,3.2
...
```

表示 图像`data/train/refimgs/carnivaldolls.bmp`的评分为3.2。
在训练时，如果配置`num_classes=1`，则损失函数计算为：||output - 3.2||；
配置`num_classes=N`，则损失函数计算为：||(output.softmax() * [1:N+1]).sum() - 3.2||；
详细可参考损失函数`criterions/regress_loss.py`。


### 部分重要配置参数说明

针对`config.py`里的部分重要参数说明如下：

- `--data`: 数据集根目录，下面包含`train`, `test`, `val`三个目录的数据集，默认当前文件夹下`data/`目录；
- `--image_size`: 输入应该为两个整数值，预训练模型的输入时正方形的，也就是[224, 224]之类的；
实际可以根据自己需要更改，数据预处理时，会将图像centercrop等比例部分，然后resize指定的输入尺寸。
- `--num_classes`: 模型的预测分支数；
- `-b`: 设置batch size大小，默认为256，可根据GPU显存设置；
- `-j`: 设置数据加载的进程数，默认为8，可根据CPU使用量设置；
- `--criterion`: 损失函数，`ranking_loss`, `emd_loss`, `regress_loss`三种；
- `--margin`: 配合`ranking_loss`使用，margin表示对比的差异间隔；
- `--lr`: 初始学习率，`main.py`里我默认使用Adam优化器；目前学习率的scheduler我使用的是`LambdaLR`接口，自定义函数规则如下，
详细可参考`main.py`的`adjust_learning_rate(epoch, args)`函数：
```
~ warmup: 0.1
~ warmup + int([1.5 * (epochs - warmup)]/4.0): 1, 
~ warmup + int([2.5 * (epochs - warmup)]/4.0): 0.1
~ warmup + int([3.5 * (epochs - warmup)]/4.0) 0.01
~ epochs: 0.001
```
- `--warmup`: warmup的迭代次数，训练前warmup个epoch会将 初始学习率*0.1 作为warmup期间的学习率；
- `--epochs`: 训练的总迭代次数；
- `--resume`: 权重文件路径，模型文件将被加载以进行模型初始化，`--jit`和`--evaluation`时需要指定；
- `--jit`: 将模型转为JIT格式，利于部署；
- `--evaluation`: 在测试集上进行模型评估；

---

## 快速使用 —— 使用公开数据集AVA(aesthetic visual analysis)进行训练、测试、部署

下载数据[mtobeiyf/ava_downloader](https://github.com/mtobeiyf/ava_downloader)

---

## 使用说明

可参考对应的`z_task_shell/*.sh`文件

### 模型信息打印

打印分支数为`6`、输入图像分辨率`400x224`的`efficientnet-b0`网络的基本信息：

```shell
python main.py --arch efficientnet_b0 --num_classes 6 --image_size 244 224
```

### 模型训练

基于`data/`目录下的`train`数据集，使用分支数为`1`、输入图像分辨率`244x224`、损失函数为排序对比损失ranking loss（+ margin=0.1） 
的`efficientnet-b0`网络，同时加载预训练模型、训练学习率warmup 5个epoch，batch size为384，
数据加载worker为16个，训练65个epoch：

```shell
python main.py --data data/ --train --arch efficientnet_b0 --num_classes 1 \
--criterion=rank --margin 0.1 -- --use_margin --image_size 244 224 \
--pretrained --warmup 5 --epochs 65 -b 384 -j 16 --gpus 1 --nodes 1
```

参数的详细说明可查看`config.py`文件。

### 模型评估

基于`data/`目录下的`test`数据集，评估`checkpoints/model_best_efficientnet-b0.pth`目录下的模型
（需要指定模型输入、输出、损失函数、模型结构，数据加载的worker，推理时的batch size）：

```shell
python main.py --data data -e --arch efficientnet_b0 --num_classes 1 --criterion=rank --margin 0.1 \
--image_size 244 224 --batch_size 5 --workers 0 --resume checkpoints/model_best_efficientnet-b0.pth -g 1 -n 1
```

参数的详细说明可查看`config.py`文件。

### 模型转换

转为jit格式模型文件：

```shell
python main.py --jit --arch efficientnet_b0 --num_classes 1 --resume checkpoints/model_best_efficientnet-b0.pth -g 0
```

### 模型部署demo

训练好模型后，想用该模型对图像数据进行预测，可使用`demos`目录下的脚本`image_assessment.py`：

```shell
cd demos
python image_assessment.py
```

---

## 数据集

| 数据集            | 介绍    | 备注                      | 网址 |
| ---------------- |:------:|:------------------------:|:--------------:|
| TID2013          | 质量评价 | 25张参考图像，24个失真类型   | [TID2013](http://www.ponomarenko.info/tid2013.htm)            | 
| LIVE             | 质量评价 | 29张参考图像，5个失真类型    | [LIVE](https://live.ece.utexas.edu/research/quality/subjective.htm) | 
| MLIVE            | 质量评价 | 15张参考图像，15个失真类型   | [MLIVE](https://live.ece.utexas.edu/research/quality/live_multidistortedimage.html) | 
| WATERLOO         | 质量评价 | 4744张参考图像，20个失真类型 | [Waterloo](https://ece.uwaterloo.ca/~k29ma/exploration/)     | 
| photo.net        | 美观评价 | 20,278张图像，打分[0,10]   | [photo.net](http://ritendra.weebly.com/aesthetics-datasets.html) |
| DPChallenge.com  | 美观评价 | 16,509张图像，打分[0,10]   | [DPChallenge](http://ritendra.weebly.com/aesthetics-datasets.html)  |
| AVA              | 美观评价 | 255,500张图像，打分[0,10]  | [ava_downloader](https://github.com/mtobeiyf/ava_downloader) | 

---

## Reference

[d-li14/mobilenetv3.pytorch](https://github.com/d-li14/mobilenetv3.pytorch)

[lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)

[zhanghang1989/ResNeSt](https://github.com/zhanghang1989/ResNeSt)

[titu1994/neural-image-assessment](https://github.com/titu1994/neural-image-assessment)

## TODO

- [x] 预训练模型下载URL整理（参考Reference）；
- [ ] 模型的openvino格式的转换和对应的部署demo；




