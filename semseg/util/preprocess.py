import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Dataset => data-list')
    parser.add_argument('--force_cover', action=store_true, default=False, help="force to re-generate data-list")
    return parser.parse_args()

def form_datalist(root, data_list, args):         
    if not os.path.isdir(root):
        raise RuntimeError("dataset not exist")
    if os.path.isfile(data_list):
        if not args.force_cover:
            print("Found existing data-list, >>skip")
            return
    else:
        f = open(data_list, "")
    
    from tqdm import tqdm
    img_paths = os.listdir(root)
    for i in tqdm(range(len(img_paths)), desc="Processed:"):
        img_path = os.path.join(root, img_paths[i])
        label_path = os.path
                                
if __name__ == "__main__":            
    root = "/home/data/1945"
    data_list = "train_data.txt"

    args = get_parser()
    form_datalist(root, data_list, args)