import os
import glob
from tqdm import tqdm

dataset_root = '/Volumes/ssd_imran/carla_dataset/training/carla_semantic/'


def create_list(state):
    dir_ = os.path.join(dataset_root, state)

    images = glob.glob(
        os.path.join(
            dir_,
            'images/*.png'
        )
    )

    images.sort()

    list_lines = []

    for _img in images:
        list_lines.append(
            f'{_img.replace(dataset_root, "")}\t{_img.replace("images", "labels").replace(dataset_root, "")}'
        )

    with open(f'/Users/imrankabir/Desktop/research/semantic_seg_audio_description/HRNet-Semantic-Segmentation-mod/data/list/carla/{state}.lst', 'w') as f:
        f.write("\n".join(list_lines))


folders = ['train', 'validation']

for fol in folders:
    create_list(fol)