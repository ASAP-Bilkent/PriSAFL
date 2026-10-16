import pickle
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

with open(DATA_DIR / 'predicted_quant.pickle', 'rb') as handle:
    predicted_quant = pickle.load(handle)
with open(DATA_DIR / 'censorship.pickle', 'rb') as handle:
    censorship = pickle.load(handle)
with open(DATA_DIR / 'survival.pickle', 'rb') as handle:
    survival = pickle.load(handle)

events = np.reshape([True if i == 1 else False for i in censorship], (384, 1))
survival = np.reshape(survival, (384, 1))

pathway_info = []
with open(DATA_DIR / "result.csv") as f:
    for index, line in enumerate(f):
        if index == 0:
            continue
        tokens = line.split(",")
        pathway_info.append(tokens[2:])

pathway_info = np.reshape(pathway_info, (37, len(pathway_info[0])))
pathway_info = pathway_info.astype(np.int64)
