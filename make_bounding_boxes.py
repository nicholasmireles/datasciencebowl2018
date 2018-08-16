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
for im, seg_ims in tqdm(train_dict.items(), total=len(train_dict.keys())):
    for seg_im in seg_ims:
        img = imread(seg_im)
        top = np.argwhere(img==255)[:,0].min()
        bottom = np.argwhere(img==255)[:,0].max()
        left = np.argwhere(img==255)[:,1].min()
        right = np.argwhere(img==255)[:,1].min()
        #bbs.append([im, seg_im, left, top, right, bottom, 'nucleus'])
        im = os.path.join(cwd, im)
        bbs.append([im, left, top, right, bottom, 'nucleus'])

df = pd.DataFrame().from_dict(bbs)
#df.columns = ['image', 'mask', 'left', 'top', 'right', 'bottom', 'class']
df.columns = ['image', 'left', 'top', 'right', 'bottom', 'class']
df.to_csv('mask_bounding_boxes.csv', index=False, header=False)
