import numpy as np
from PIL import Image

def visualize(label_img, output_path):
    color_map = {
        # tuple means R, G. B
        0: (0, 0, 0), # background
        1: (105, 119, 35), # algae
        2: (112, 6, 20), # dead_twigs_leaves
        3: (147, 112, 219), # rubbish
        4: (230, 153, 102), # water
    }

    h, w = label_img.shape[:2]
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for i, c in color_map.items():
        color_img[label_img == i, :] = c
    Image.fromarray(color_img).save(output_path)
