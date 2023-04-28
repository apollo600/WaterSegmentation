# split My dataset to train.txt, val.txt and trainval.txt

import os
import random
from shutil import copyfile


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
                

def build_dir_structure(data_path, root_path):
                                                    
    """
    .
    ├── ImageSets
    ├── JPEGImages
    ├── SegmentationClass
    """
    
    # Make Directories
    os.makedirs(
        os.path.join(root_path, "ImageSets", "Segmentation"), exist_ok=True
    )
    os.makedirs(
        os.path.join(root_path, "JPEGImages"), exist_ok=True
    )
    os.makedirs(
        os.path.join(root_path, "SegmentationClass"), exist_ok=True
    )

    file_list = os.listdir(data_path)
    file_list = [ x[:-4] for x in file_list if x.endswith('.jpg') ]
    
    for x in file_list:
        copyfile(
            os.path.join(data_path, x + ".jpg"),
            os.path.join(root_path, "JPEGImages", x + ".jpg")
        )
        copyfile(
            os.path.join(data_path, x + ".png"),
            os.path.join(root_path, "SegmentationClass", x + ".png")
        )
    
    split(file_list, 0.9, os.path.join(root_path, "ImageSets", "Segmentation"))

    os.system(f"tree -d {root_path}")
    

if __name__ == "__main__":            
    build_dir_structure("/home/data/1945", "/project/train/src_repo/MyDataset")