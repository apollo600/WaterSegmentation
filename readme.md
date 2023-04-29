# Water Segmentation Competition

The repo is for [this competition](https://www.cvmart.net/topList/10488).

## Usage

Environment (model1): pytorch, numpy, tqdm.
Environment (model2): pytorch, numpy, tqdm, tensorboard, opencv-python.

Just following the command below.

### Linux

Method1 (UNet)

```shell
# change the train.sh and inference.sh path
bash ./UNet/train.sh [dataset-name]
bash ./UNet/inference.sh
```

Method2 (DeepLabV3+)

```shell
# change the train.sh and inference.sh path
bash ./DeepLab/train.sh
python ./DeepLab/inference.py
```

### Windows

Method1 (UNet)

```shell
cd $THIS_REPO_NAME
UNet\train [dataset-name]
UNet\inference
```

Method2 (DeepLabV3+)

```shell
cd $THIS_REPO_NAME
DeepLab\train
DeepLab\inference
```

The `[dataset-name]` should be `Kitti` or `My`, default is `My`.

### PS

- Note 1: Root paths in Windows and Linux are different. Change `*.bat` or `*.sh` file if you need.
- Note 2: `inference` requires you to specify the model path in the `inference.bat` or `inference.sh` or `inference.py` file.

### Competition Upload

[This url again](https://www.cvmart.net/topList/10488) if you need to refer.

```shell
cd /project/train/
rm -rf src_repo
git clone $THIS_REPO
mv $THIS_REPO_NAME src_repo
...  # test or train the repo as you want
<Train it>  # command: bash /project/train/src_repo/DeepLab/train.sh
...  # wait till your training is done...
cp DeepLab/ji.py /project/ev_sdk/  # copy the interface
vim /project/ev_sdk/ji.py  # to specify the model path in the file
<Test it>
DONE~
```

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
