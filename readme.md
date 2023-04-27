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
├─ train.py  <-- usual train function
├─ ji.py     <-- platform train interface
└─ *.sh / *.bat      <-- quick access
```

# Mannual Interface

```python
def init() -> HANDLE:
    return HANDLE()

def process_image(handle: HANDLE, input_image=None, args=None, **kwargs) ->
    args = json.loads(args)
    mask_output_path = args['mask_output_path']
    # Process image here
    # Generate dummy mask data
    h, w, _ = input_image.shape
    dummy_data = np.random.randint(low=0, high=2, size=(w, h), dtype=np.uint8)
    pred_mask_per_frame = Image.fromarray(dummy_data)
    pred_mask_per_frame.save(mask_output_path)
    return json.dumps({'mask': mask_output_path}, indent=4)
```

## Todo

- [ ] ??
- [ ] ??
