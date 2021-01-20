import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import preprocess, CheXpert
from densenet import train, densenet121
from config import opt
from CBR import *

if __name__ == "__main__":
	
	RootPath = "D:/project/shallowNN/dataset/"
	df_train = pd.read_csv(RootPath + "CheXpert-v1.0-small/TRAIN.csv")
	df_train = preprocess(df_train, RootPath)
	
	df_val = pd.read_csv(RootPath + "CheXpert-v1.0-small/VAL.csv")
	df_val = preprocess(df_val, RootPath)

	df_train = df_train.sample(5000)
	df_val = df_val.sample(5000)
	

	index = [0, 2, 3]
	df_train = df_train.iloc[:, index]
	df_val = df_val.iloc[:, index]

	# print(df_train.head())
	# print(df_train.columns)
	# asf

	train_dl = DataLoader(CheXpert(df_train), batch_size=opt.bs, shuffle=True)
	val_dl = DataLoader(CheXpert(df_val), batch_size=opt.bs, shuffle=False)


	model = CBR_Tiny(output_class=df_train.shape[1]-1, input_channels=3, binary=True)
	# model = densenet121(num_classes=df_train.shape[1]-1, pretrained=False)

	train(model, train_dl, val_dl, use_gpu=True)	
	
	


