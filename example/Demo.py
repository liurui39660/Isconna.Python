from os import environ
from pathlib import Path

from pyprojroot import here
from sklearn.metrics import roc_auc_score
from tqdm import tqdm, trange

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
	# I can process them streamingly, but I/O is too slow

	n = int(pathMeta.read_bytes())

	src = [0] * n
	dst = [0] * n
	ts = [0] * n
	data = pathData.read_bytes().splitlines()[:n]
	for i in trange(n, desc="Load Dataset", unit_scale=True):
		src[i], dst[i], ts[i] = map(int, data[i].split(b","))

	label = list(map(int, pathLabel.read_bytes().splitlines()[:n]))

	# Do the magic
	# --------------------------------------------------------------------------------

	score = [0.0] * n
	isc = Isconna.EdgeOnlyCore(shapeCMS[0], shapeCMS[1], zeta)
	# isc = Isconna.EdgeNodeCore(shapeCMS[0], shapeCMS[1], zeta)
	for i in trange(n, desc=isc.nameAlg, unit_scale=True):
		score[i] = isc.Call(src[i], dst[i], ts[i], alpha, beta, gamma)

	# Export raw scores
	# --------------------------------------------------------------------------------

	pathScore.parent.mkdir(exist_ok=True)
	with open(pathScore, "w", newline="\n") as file:
		for sc in tqdm(score, "Export Scores", unit_scale=True):
			file.write(f"{sc}\n")
	print(f"// Raw scores are exported to\n// {pathScore}")

	# Evaluate results
	# --------------------------------------------------------------------------------

	print(f"ROC-AUC = {roc_auc_score(label, score):.4f}")
