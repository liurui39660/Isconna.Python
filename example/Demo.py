from mmap import ACCESS_READ, mmap
from os import environ
from pathlib import Path

from pyprojroot import here
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import Isconna

if __name__ == '__main__':
	dirDataset = Path(environ.get("DATASET_DIR", here() / "data"))

	# Parameter
	# --------------------------------------------------------------------------------

	# pathMeta = dirDataset / "DARPA/processed/Meta.txt"
	# pathData = dirDataset / "DARPA/processed/Data.csv"
	# pathLabel = dirDataset / "DARPA/processed/Label.csv"

	pathMeta = dirDataset / "CIC-IDS2018/processed/Meta.txt"
	pathData = dirDataset / "CIC-IDS2018/processed/Data.csv"
	pathLabel = dirDataset / "CIC-IDS2018/processed/Label.csv"

	# pathMeta = dirDataset / "UNSW-NB15/processed/Meta.txt"
	# pathData = dirDataset / "UNSW-NB15/processed/Data.csv"
	# pathLabel = dirDataset / "UNSW-NB15/processed/Label.csv"

	# pathMeta = dirDataset / "ISCX-IDS2012/processed/Meta.txt"
	# pathData = dirDataset / "ISCX-IDS2012/processed/Data.csv"
	# pathLabel = dirDataset / "ISCX-IDS2012/processed/Label.csv"

	# pathMeta = dirDataset / "CTU-13/processed/Meta.txt"
	# pathData = dirDataset / "CTU-13/processed/Data.csv"
	# pathLabel = dirDataset / "CTU-13/processed/Label.csv"

	# pathMeta = dirDataset / "CIC-DDoS2019/processed/Meta.txt"
	# pathData = dirDataset / "CIC-DDoS2019/processed/Data.csv"
	# pathLabel = dirDataset / "CIC-DDoS2019/processed/Label.csv"

	pathScore = here() / "out/Score.txt"

	alpha = 1.0
	beta = 1.0
	gamma = 0.5
	zeta = 0.7
	shapeCMS = [2, 3000]

	# Read dataset
	# --------------------------------------------------------------------------------

	with open(pathMeta) as fileMeta:
		n = int(fileMeta.readline())

	src = [0] * n
	dst = [0] * n
	ts = [0] * n
	with open(pathData) as fileData, mmap(fileData.fileno(), 0, access=ACCESS_READ) as mmapData:  # type: mmap
		for i, edge in tqdm(enumerate(map(lambda a: map(int, a.split(b",")), mmapData.read().split(b"\n", n)[:n])), "Loading data", n, unit_scale=True):
			src[i], dst[i], ts[i] = edge  # I know I can use zip(), but this is more interactive

	with open(pathLabel) as fileLabel, mmap(fileLabel.fileno(), 0, access=ACCESS_READ) as mmapLabel:  # type: mmap
		label = list(map(int, mmapLabel.read().split(b"\n", n)[:n]))

	# Do the magic
	# --------------------------------------------------------------------------------

	score = [0.0] * n
	isc = Isconna.EdgeOnlyCore(shapeCMS[0], shapeCMS[1], zeta)
	# isc = Isconna.EdgeNodeCore(shapeCMS[0], shapeCMS[1], zeta)
	for i in tqdm(range(n), isc.nameAlg, unit_scale=True):
		score[i] = isc.FitPredict(src[i], dst[i], ts[i], alpha, beta, gamma)

	# Export raw scores
	# --------------------------------------------------------------------------------

	# with open(pathScore, "w", newline="\n") as fileScore:
	# 	for sc in tqdm(score, "Exporting scores", unit_scale=True):
	# 		fileScore.write(f"{sc}\n")
	# print(f"// Raw scores are exported to\n// {pathScore}")

	# Evaluate results
	# --------------------------------------------------------------------------------

	print(f"ROC-AUC = {roc_auc_score(label, score):.4f}")
