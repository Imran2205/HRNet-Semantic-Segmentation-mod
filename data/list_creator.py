import os
import glob
from tqdm import tqdm

dataset_root = '/Users/imrankabir/Downloads/pfb/'


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
            f'{_img.replace(dataset_root, "")}\t{_img.replace("images", "labels").replace("jpg", "png").replace(dataset_root, "")}'
        )

    with open(f'./list/pfb/{state}.lst', 'w') as f:
        f.write("\n".join(list_lines))


# folders = ['train', 'validation', 'test']
folders = ['test']

for fol in folders:
    create_list(fol)