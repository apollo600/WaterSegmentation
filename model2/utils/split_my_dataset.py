# split My dataset to train.txt, val.txt and trainval.txt

import os
import random


def split(file_list, train_percentage=0.9, output_path=""):           
        
    total_size = len(file_list)
    train_size = int(total_size * train_percentage)
    val_size = total_size - train_size

    train_sample = random.sample(range(total_size), train_size)
    val_sample = [ x for x in range(total_size) if x not in train_sample ]

    assert(output_path != "")
    os.makedirs(output_path, exist_ok=True)
    
    with open(os.path.join(output_path, "trainval.txt"), "w") as f:
        for i in range(total_size):
            f.write(f"{file_list[i]}\n")

    with open(os.path.join(output_path, "train.txt"), "w") as f:
        for i in train_sample:
            f.write(f"{file_list[i]}\n")

    with open(os.path.join(output_path, "val.txt"), "w") as f:
        for i in val_sample:
            f.write(f"{file_list[i]}\n")
                

def build_dir_structure(root_path):
    
    
    


if __name__ == "__main__":            
    file_list = os.listdir("/home/data/1945")
    file_list = [ x[:-4] for x in file_list if x.endswith('.jpg') ]
    
    split(file_list, 0.9, "/project/train/src_repo/MyImageSets")