# EAST-Pytorch
EAST的Pytorch版本, paper:https://arxiv.org/abs/1704.03155

## Thanks
- [argman/EAST](https://github.com/argman/EAST)
- [songdejia/EAST](https://github.com/songdejia/EAST)

## Requirements
- Python3
- Pytorch > 0.4.0
- numpy
- shapely
- Polygon3(for evaluation)

## Dataset
labelx format
```
{
  "url": "/path/to/your/images.jpg",
  "type": "image",
  "invalid": false,
  "label": [
    {
      "name": "outline",
      "type": "detection",
      "version": "1",
      "data": [
        {
          "bbox": [
            [
              662,
              76
            ],
            [
              845,
              76
            ],
            [
              845,
              162
            ],
            [
              662,
              162
            ]
          ],
          "class": "a"
        }
      ]
    }
  ]
}
```

## Train
### 1. Modify config.py
```
class Args:
    train_dir = None
    train_annotation = '/path/to/your/trainset.json' # 训练数据的labelx标签
    val_dir = None
    val_annotation = '/path/to/your/valset.json' # 验证数据的labelx标签
                                                   每训练eval_interval个epoch, 验证一次, 保存验证F1最高的checkpoint
                                                   若指定为None, 训练过程中不验证
    backbone = 'resnet50' # resnet50 or resnet18
    batch_size = 4
    lr = 0.00001
    weight_decay = 1e-5
    epochs = 200
    warm_up = 100
    num_workers = 4
    input_size = 512
    text_scale = 512
    inference_size = (672, 480) # int或tuple(w, h), int时等比例缩放并限制长边尺寸<inference_size, tuple时, resize成inference_size
    min_text_size = 10 # 训练过程中跳过 < min_text_size的文本
    gpus = "0" # gpu ids
    backbone_pretrain = False # 是否load backbone的pretrain model
                                仅从头开始训练时指定为True
    checkpoint_dir = './checkpoint' # 指定模型保存目录
    checkpoint = './checkpoint/checkpoint_best.path.tar' # 从checkpoint恢复训练模型, 中断后继续训练或finetune时需指定
                                                           若从头开始训练, 指定为None
    finetune = True # finetune from checkpoint
                      reset optimizer if True else restore optimizer from checkpoint
    print_freq = 2
    eval_interval = 1 # eval interval (epoch)
    tensorboardX = None
```
训练有三种情况:
1. 从头训练
2. 从checkpoint恢复训练
3. 从checkpoint finetune   

2和3的区别是: 2会忽略config.py中的optimizer参数, 使用checkpoint中保存的optimizer参数; 3会根据config.py重置optimize参数.

### 2.make geo_map_cython_lib
```
cd data/geo_map_cython_lib
sh build.sh
```

### 3. train
```
➜ python3 train.py
Using 1 GPUs...
...
06/17 03:55:48  Train Epoch [1][1364/1513] Batch Time 0.557(0.553) Data Time 0.011(0.013) Loss 0.010(0.013) Lr 0.000100
06/17 03:55:49  Train Epoch [1][1366/1513] Batch Time 0.558(0.553) Data Time 0.010(0.013) Loss 0.019(0.013) Lr 0.000100
...
```


## Inference
```
$ python3 inference.py -h
make: Entering directory '/workspace/mnt/group/algorithm/pengyuanzhuo/east/lanms'
make: 'adaptor.so' is up to date.
make: Leaving directory '/workspace/mnt/group/algorithm/pengyuanzhuo/east/lanms'
usage: EAST detect [-h] [--draw DRAW] [--backbone {resnet50,resnet18}] [--text_scale TEXT_SCALE]
                   [--input_size INPUT_SIZE [INPUT_SIZE ...]] [--score_map_thresh SCORE_MAP_THRESH] [--box_thresh BOX_THRESH]
                   [--nms_thresh NMS_THRESH]
                   PTH DIR RES

positional arguments:
  PTH                   model path
  DIR                   image dir
  RES                   result file

optional arguments:
  -h, --help            show this help message and exit
  --draw DRAW           draw dir
  --backbone {resnet50,resnet18}
                        backbone, resnet50 or resnet18, default=resnet50
  --text_scale TEXT_SCALE
                        text scale, default=512
  --input_size INPUT_SIZE [INPUT_SIZE ...]
                        input image size, int(max side) or tuple(w, h), default=768.
  --score_map_thresh SCORE_MAP_THRESH
                        threshold for score map, default=0.8
  --box_thresh BOX_THRESH
                        threshold for boxes, default=0.1
  --nms_thresh NMS_THRESH
                        nms threshold, default=0.2
```                        
指定draw参数可以绘制检测结果


## Eval
- groundtruth: gt.json
- 推理结果: res.json
```
cd pyicdartools
python evaluation.py -g=/path/to/gt.json -s=/path/to/res.json
```
=> recall, precision, F1 score

