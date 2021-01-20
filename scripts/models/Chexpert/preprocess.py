import numpy as np
import pandas as pd
from skimage import io
from joblib import Parallel, delayed
from skimage.transform import resize
import os

RootPath = "/home/jusun/shared/Stanford_Dataset/"

def read_img(p):
	'''
	if os.path.exists(p):
		return 1
	else:
		return 0
	'''
	
	img = io.imread(p)
	w, h = img.shape
	if w > h:
		pad = (w-h)//2
	elif w < h:
		pad = (h-w)//2
	else:
		pass

	img = resize(img, (224, 224))
	return img

def load_img():
	df = pd.read_csv(RootPath + "CheXpert-v1.0-small/train.csv")
	df = df.sample(10)
	path = RootPath + df['Path']
	results = Parallel(n_jobs=-1, backend="loky")(delayed(read_img)(p) for p in path)
	results = np.asarray(results)
	print(results.shape)

	# load labels
	columns = df.columns[5:-1]
	df = df[columns].fillna(0)
	df = df.replace(-1, 1)
	
	print(columns)


	return results

if __name__ == "__main__":
	load_img()
