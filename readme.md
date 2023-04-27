# Water Segmentation

## Usage

Note 1: The `[dataset-name]` below should be `Kitti` or `My`, default is `My`.  
Note 2: Paths in Windows and Linux are different. Change if you need.

### Windows

```shell
train [dataset-name]
```

### Linux

```shell
chmod 775 train.sh
./train.sh [dataset-name]
```

## Architecture

```plaintext
WaterSegmentation
├─ model/    <-- model and loss classes
├─ utils/    <-- dataset reader
├─ train.py  <-- main train function
└─ *.sh / *.bat      <-- quick access
```

## Todo

- [ ] ??
- [ ] ??
