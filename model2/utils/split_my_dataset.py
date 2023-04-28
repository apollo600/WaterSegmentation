# split My dataset to train.txt, val.txt and trainval.txt

import os
import random


def split(file_list, train_percentage=0.9, output_path=""):           
                                                    
    total_size = len(file_list)
    train_size = int(total_size * train_percentage)
    val_size = total_size - val_size

    train_sample = random.sample(total_size, train_size)
    val_sample = [ x for x in range(total_size) if x not in train_sample ]

    assert(output_path != "")
    with open(os.path.join(output_path, )) as f:
                
    

if __name__ == "__main__":            
    file_list = os.listdir("/home/data/1945")
    file_list = [ x[:-4] for x in file_list if x.endswith('.jpg') ]
    
    split(file_list, 0.9, "/project/train/src_repo/MyImageSets")