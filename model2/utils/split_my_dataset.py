# split My dataset to train.txt, val.txt and trainval.txt

import os


def split(file_list, train_p output_path):           
                    
    total_size = len(file_list)



if __name__ == "__main__":            
    file_list = os.listdir("/home/data/1945")
    file_list = [ x[:-4] for x in file_list if x.endswith('.jpg') ]
    
    split(file_list, "/project/train/src_repo/MyImageSets")