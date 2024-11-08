
from jiwer import wer
from transformers import Wav2Vec2FeatureExtractor
import torch, json, os, librosa, transformers, gc
import torch.nn as nn
import json 
import torch.nn.functional as F
from pyctcdecode import build_ctcdecoder
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import torch
import einops
from dataloader import text_to_tensor
from gcn_model import GCN_MDD
from jiwer import wer
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

random.seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCN_MDD.from_pretrained(
    'facebook/wav2vec2-base-100h', 
)
ckp = torch.load("./checkpoint/gcn_mdd.pth")
model.load_state_dict(ckp)
model = model.to(device)

phoneme_embeddings = model.get_embedding().detach().cpu().numpy()
char = ['t', 'uw', 'er', 'ah', 'sh', 'ng', 'ow', 'aw', 'aa', 'th', 'ih', 'zh', 'k', 'y', 'l', 'uh', 'ch', 'w', 'b', 'v', 'ao', 's', 'p', 'iy', 'r', 'eh', 'f', 'n', 'ay', 'oy', 'd', 'g', 'ey', 'err', 'dh', 'ae', 'hh', 'm', 'jh', 'z', '']
char = list(json.load(open("category.json", "r")).values())
tsne = TSNE(n_components=2, perplexity=10)
phoneme_embeddings_2d = tsne.fit_transform(phoneme_embeddings)
plt.figure(figsize=(10, 8))
plt.scatter(phoneme_embeddings_2d[:, 0], phoneme_embeddings_2d[:, 1], color='red')
for i in range(len(phoneme_embeddings_2d)):
    plt.text(phoneme_embeddings_2d[i, 0], phoneme_embeddings_2d[i, 1], char[i], fontsize=9)

plt.title("t-SNE Visualization of Phoneme Embeddings")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.savefig("embedding.png", format='png')
plt.close()


phoneme_embeddings = model.get_look_up_table().detach().cpu().numpy()
tsne = TSNE(n_components=2, perplexity=10)
phoneme_embeddings_2d = tsne.fit_transform(phoneme_embeddings)
plt.figure(figsize=(10, 8))
plt.scatter(phoneme_embeddings_2d[:, 0], phoneme_embeddings_2d[:, 1], color='red')
for i in range(len(phoneme_embeddings_2d)):
    plt.text(phoneme_embeddings_2d[i, 0], phoneme_embeddings_2d[i, 1], char[i], fontsize=9)

plt.title("t-SNE Visualization of Phoneme Embeddings")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.savefig("graph.png", format='png')
plt.close()