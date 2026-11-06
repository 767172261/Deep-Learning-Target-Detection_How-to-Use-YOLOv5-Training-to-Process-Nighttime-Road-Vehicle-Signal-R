深度学习目标检测中_如何使用YOLOv5训练处理夜间道路车辆信号灯识别检测数据集并对车辆信号灯做识别检测任务，实现可视化评估及推理
@[toc]

夜间车辆 信号灯识别检测数据集VOC 3500张 
YOLO数据集
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/24a460a548804a4fb460e2f827841a6f.png)

| 标注名称 | 数量 |
| ---- | ---- |
| bicycle | 178 |
| bus | 297 |
| car | 30521 |
| motorcycle | 86 |
| other vehicle | 6 |
| pedestrian | 2392 |
| rider | 103 |
| traffic light | 10515 |
| traffic sign | 11042 |
| train | 5 |
| truck | 615 |
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e084fad5e5a54811a2d1745dd1a343c7.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ea1108d84c254016b06d2cf779211e78.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9720afe7c2434bb396d9dd5c770d0390.png)
当前YOLO的最新版本是YOLOv5，而截至我的知识更新日期（2024年），并没有名为“YOLOv11”的版本发布。因此，我将基于YOLOv5来继续您的请求，即使用夜间车辆信号灯识别检测数据集进行模型训练、评估和推理。

### 使用YOLOv5处理夜间车辆信号灯识别检测任务

#### 1. 环境准备

首先确保安装了必要的依赖项：

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

#### 2. 数据准备

根据您提供的信息，数据集为VOC格式并已转换为YOLO格式。我们需要创建一个`data.yaml`文件来定义数据集路径和类别信息。

##### 创建`data.yaml`文件：

```yaml
train: ./path/to/train/images
val: ./path/to/val/images

nc: 11  # 类别数量
names: ['bicycle', 'bus', 'car', 'motorcycle', 'other_vehicle', 'pedestrian', 'rider', 'traffic_light', 'traffic_sign', 'train', 'truck']
```

请确保替换上述路径为您实际的数据集路径。

#### 3. 划分数据集

如果尚未划分训练集和验证集，可以使用以下脚本进行划分：

```python
import os
import random
import shutil

# 设置路径
dataset_dir = './path/to/dataset'
train_dir = './path/to/train'
val_dir = './path/to/val'
split_ratio = 0.8

# 创建目录
os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)

# 获取所有图片文件名
image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]
random.shuffle(image_files)

# 划分数据集
split_idx = int(len(image_files) * split_ratio)
for i, image_file in enumerate(image_files):
    src_image = os.path.join(dataset_dir, image_file)
    src_label = os.path.join(dataset_dir, image_file.replace('.jpg', '.txt'))
    
    if i < split_idx:
        dst_dir = train_dir
    else:
        dst_dir = val_dir
    
    shutil.copy(src_image, os.path.join(dst_dir, 'images', image_file))
    shutil.copy(src_label, os.path.join(dst_dir, 'labels', image_file.replace('.jpg', '.txt')))
```

#### 4. 模型训练

使用YOLOv5进行训练：

```bash
python train.py --img 640 --batch 16 --epochs 100 --data path/to/data.yaml --weights yolov5s.pt
```

#### 5. 评估

训练完成后，可以使用以下命令对验证集进行评估并生成预测结果的可视化：

```bash
python val.py --weights runs/train/exp/weights/best.pt --data path/to/data.yaml --img 640 --task val
```

这将在指定的输出目录中生成带有边界框的图像，以及每个检测结果的置信度分数。

#### 6. 推理

对于新的图片或视频流，你可以使用类似的命令进行推理：

```bash
python detect.py --weights runs/train/exp/weights/best.pt --source ./path/to/new/image.jpg --conf-thres 0.5 --save-txt --save-conf
```

### 完整的操作流程总结

1. **环境搭建**：安装必要的依赖。
2. **数据准备**：确保数据集按要求组织，并创建`data.yaml`文件。
3. **数据集划分**：如有必要，使用脚本划分训练集和验证集。
4. **模型训练**：使用YOLOv5进行模型训练。
5. **评估**：在验证集上评估模型性能。
6. **推理**：使用训练好的模型对新数据进行推理。


