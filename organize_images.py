import os
import json
import shutil
from glob import glob


subdirs = os.listdir('stage1_train')
train_dict = {}

for sd in subdirs:
	if sd != '.DS_Store':
		im = glob(os.path.join('stage1_train', sd, 'images', '*.png'))
		masks = glob(os.path.join('stage1_train', sd, 'masks', '*.png'))
		train_dict[im[0]] = masks

json_dict = json.dumps(train_dict)
f = open('train_map.json', 'w+')
f.write(json_dict)
f.close()