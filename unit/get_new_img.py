import os
import shutil
from tqdm import tqdm

path_old = '../rainy_image_dataset/testing/ground_truth_old/'
path_new = '../rainy_image_dataset/testing/ground_truth/'

img_names = os.listdir(path_old)
pbar = tqdm(total=len(img_names))
for img_name in img_names:
    pbar.update(1)
    for num in range(1,15):
        shutil.copy(path_old + img_name,
                    path_new + img_name.replace('.jpg','') + '_' + str(num) + '.jpg')

pbar.close()


