import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib


from dataset import preprocess, CheXpert
from densenet import train, densenet121
from config import opt

def setup():
	SIZE = 20
	matplotlib.rc('font', size=SIZE)
	matplotlib.rc('axes', titlesize=SIZE)

if __name__ == "__main__":
	setup()
	torch.cuda.empty_cache()
	
	RootPath = "D:/project/shallowNN/dataset/"
	df = pd.read_csv(RootPath + "CheXpert-v1.0-small/TEST.csv")
	df = preprocess(df, RootPath)

	# df = df.sample(100)

	test_dl = DataLoader(CheXpert(df), batch_size=20, shuffle=True)

	# model_f = torch.load("D:/project/shallowNN/ShallowDeepNN/scripts/models/Chexpert/checkpoints/m_0120_012145.pth.tar", 
	# 	map_location=torch.device("cpu"))
	# model_f = model_f['state_dict']
	model_f = torch.load("D:/project/shallowNN/ShallowDeepNN/scripts/models/Chexpert/checkpoints/epoch_31.pth")

	model = densenet121(num_classes=df.shape[1]-1)
	model = torch.nn.DataParallel(model)
	model.load_state_dict(model_f)

	LABELS = []
	OUTPUTS = []
	model.eval()
	for d, l in tqdm(test_dl):
		print(d.shape)
		o = model(d)
		LABELS.extend(l.detach().cpu().numpy())
		OUTPUTS.extend(o.detach().cpu().numpy())

	LABELS = np.asarray(LABELS)
	OUTPUTS = np.asarray(OUTPUTS)

	print(LABELS.shape, OUTPUTS.shape)

	x = []
	y_prc = []
	y_roc = []
	for i in range(df.shape[1]-1):
		p, r, t = precision_recall_curve(LABELS[:, i], OUTPUTS[:, i])
		prc = auc(r, p)
		print("auprc for {}: {}".format(df.columns[i+1], prc))
		x.append(df.columns[i+1])
		y_prc.append(prc)

		roc = roc_auc_score(LABELS[:, i], OUTPUTS[:, i])
		y_roc.append(roc)

	plt.figure(figsize=(40, 10))
	plt.bar(x, y_prc)
	plt.xlabel("classes")
	plt.ylabel("auprc")
	plt.savefig("prc.png", dpi=300)

	plt.figure(figsize=(40, 10))
	plt.bar(x, y_roc)
	plt.xlabel("classes")
	plt.ylabel("auprc")
	plt.savefig("roc.png", dpi=300)

	


