# Water Segmentation Competition

The repo is for [this competition](https://www.cvmart.net/topList/10488).

## Usage

Environment: pytorch, numpy, tqdm. No more packages.

Just following the command below.

### Windows

```shell
train [dataset-name]
inference
```

### Linux

```shell
chmod 775 train.sh inference.sh
./train.sh [dataset-name]
./inference.sh
```

### Competition Upload

[This url again](https://www.cvmart.net/topList/10488) if you need to refer.

```shell
cd /project/train/
rm -rf src_repo
git clone $THIS_REPO
mv $THIS_REPO_NAME src_repo
...  # test or train the repo as you want
<Train it>  # command: bash /project/train/src_repo/train.sh
...  # wait till your training is done...
cp ji.py /project/ev_sdk/  # copy the interface
vim ji.py  # to specify the model path in the file
<Test it>
DONE~
```

### PS

- Note 1: The `[dataset-name]` below should be `Kitti` or `My`, default is `My`.
- Note 2: Root paths in Windows and Linux are different. Change `*.bat` or `*.sh` file if you need.
- Note 3: `inference` requires you to specify the model path in the `inference.bat` or `inference.sh` file.

## Architecture

```plaintext
WaterSegmentation
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
