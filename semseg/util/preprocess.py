import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Dataset => datalist')
    parser.add_argument('--force_cover', action=store_true, default=False, help="force to re-generate ")
    
root = "/home/data/1945"
data_list = "train_data.txt"
