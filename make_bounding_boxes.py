import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread

f = open('train_map.json')
f_raw = f.read()
f.close()

cwd = os.getcwd()

bbs = []
train_dict = json.loads(f_raw)
for im, seg_ims in tqdm(train_dict.items()):
    for seg_im in seg_ims:
        img = imread(seg_im)
        top = np.argwhere(img==255)[:,0].min()
        bottom = np.argwhere(img==255)[:,0].max()
        left = np.argwhere(img==255)[:,1].min()
        right = np.argwhere(img==255)[:,1].max()
        if top >= bottom or left >= right:
            continue
        im = os.path.join(cwd, im)
        seg_im = os.path.join(cwd, seg_im)
        bbs.append([im, left, top, right, bottom, 'nucleus', seg_im])

df = pd.DataFrame().from_dict(bbs)
df.to_csv('mask_bounding_boxes.csv', index=False, header=False)
