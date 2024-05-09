# Water Segmentation Competition

The repo is for [this competition](https://www.cvmart.net/topList/10488).

Docs:
- [水务分割识别汇报 slides](./docs/机器学习第二次作业展示.pdf)

## Usage

Environment (UNet): pytorch, numpy, tqdm.  
Environment (DeepLabV3+): pytorch, numpy, tqdm, tensorboard, opencv-python.

Just following the command below.

### Linux (cvmart.net competition)

setup see [here](#competition-setup-and-usage)

#### UNet

```shell
bash /project/train/src_repo/UNet/train.sh
bash /project/train/src_repo/UNet/inference.sh
```

#### DeepLabV3+

```shell
bash /project/train/src_repo/DeepLab/train.sh
bash /project/train/src_repo/DeepLab/inference.sh
```

### Linux

#### UNet

```shell
# you must change the path in train.sh and inference.sh
# you can refer to *.bat, or just change randomly it as you want
cd $THIS_REPO_DIR_NAME
bash ./UNet/train.sh [dataset-name]
bash ./UNet/inference.sh
```

#### DeepLabV3+

```shell
# you must change the path in train.sh and inference.sh
# you can refer to *.bat, or just change randomly it as you want
cd $THIS_REPO_DIR_NAME
bash ./DeepLab/train.sh
python ./DeepLab/inference.py
```

### Windows

#### UNet

```shell
cd %THIS_REPO_DIR_NAME%
UNet\train [dataset-name]
UNet\inference
```

The `[dataset-name]` should be `Kitti` or `My`, default is `My`.

#### DeepLabV3+

```shell
cd %THIS_REPO_DIR_NAME%
DeepLab\train
DeepLab\inference
```

## Competition Setup and Usage

[This url again](https://www.cvmart.net/topList/10488) if you need to refer.

### Download

```shell
cd /project/train/
rm -rf src_repo
git clone $THIS_REPO_GIT_OR_HTTP
mv $THIS_REPO_DIR_NAME src_repo
...  # test or train the repo as you want
```

### Train

- go to `https://www.cvmart.net/dev/10488/modelDevelopment/train`
- click `新建训练任务`  
- set `执行命令` to `bash /project/train/src_repo/DeepLab/train.sh`  
- do not mark any tick in `预加载模型`
- click `提交`
- wait till your training is done

### Test

- copy the interface by using such as  
  `mkdir -p /project/ev_sdk/src/`  
  `cp /project/train/src_repo/DeepLab/ji.py /project/ev_sdk/src/`  
- specify the model path in file `ji.py`  
- go to `https://www.cvmart.net/dev/10488/modelDevelopment/test`
- click `发起模型测试`  
- click `请选择模型列表`, choose the file you trained in the step above  
  make sure the model path in ji.py is the same as it here
- click `提交`
- wait till your testing is done

## Architecture

```plaintext
WaterSegmentation
UNet
    ├─ docs/         <-- some documents
    ├─ model/        <-- model and loss classes
    ├─ utils/        <-- dataset reader, visualization, ...
    ├─ train.py      <-- train function
    ├─ inference.py  <-- inference function
    ├─ ji.py         <-- inference interface
    └─ *.sh / *.bat  <-- quick access
DeepLab
    ├─ docs/         <-- some documents
    ├─ model/        <-- model and loss classes
    ├─ utils/        <-- dataset reader, visualization, ...
    ├─ train.py      <-- train function
    ├─ inference.py  <-- inference function
    ├─ ji.py         <-- inference interface
    └─ *.sh / *.bat  <-- quick access
```

## Interface

Refer to [this](https://www.cvmart.net/topList/10488?tab=RankDescription).

```python
def init() -> nn.Module:
def process_image(
    handle: nn.Module, input_image: np.ndarray,
    args: Any, **kwargs
) -> str
```

## Todo

See Issues.
