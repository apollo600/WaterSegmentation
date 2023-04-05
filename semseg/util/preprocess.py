import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Dataset => data-list')
    parser.add_argument('--force_cover', action=store_true, default=False, help="force to re-generate data-list")
    return parser.parse_args()

def form_datalist(root, data_list):         
    if not os.path.isdir(root):
        raise RuntimeError("dataset not exist")
    if os.path.isfile(data_list):
        print("Found existing data-list, >>skip")
        return
    else:
        f = open(data_list, "w")
    
    from tqdm import tqdm
    img_paths = os.listdir(root).sort()
    for i in tqdm(range(len(img_paths) // 2), desc="Processed:"):
        img_path = os.path.join(root, img_paths[2*i])
        label_path = os.path.join(root, img_paths[2*i+1])
        if img_path.split('.')[-1] != 'jpg' or label_path.split('.')[-1] != 'png':
            raise RuntimeError(f"Wrong file with {img_path}, {label_path}")
        f.write(f"{img_path} {label_path}\n")
    
    f.close()
            
                                
if __name__ == "__main__":            
    root = "/home/data/1945"
    data_list = "train_data.txt"

    args = get_parser()
    form_datalist(root, data_list, args)