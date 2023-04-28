import numpy as np
from PIL import Image

def visualize(label_img, output_path):
    color_map = {
        # tuple means R, G. B
        0: (0, 0, 0), # background: black
        1: (255, 255, 255), # algae: white
        2: (56, 94, 15), # dead_twigs_leaves: green
        3: (255, 97, 3), # rubbish: orange
        4: (135, 206, 235), # water: blue
    }

    h, w = label_img.shape[:2]
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for i, c in color_map.items():
        color_img[label_img == i, :] = c
    Image.fromarray(color_img).save(output_path)


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)
