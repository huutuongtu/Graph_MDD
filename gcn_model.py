import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GCN

from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model
import einops
import numpy as np
import json


f = json.load(open("./category.json", "r"))
keys = list(f.keys())
values = []
tokens = []
for char in keys:
    if "*" not in char:
        tokens.append(char)
        values.append(f[char])
classes = values 
N = len(classes)
pairs = []
for i in range(N):
    for j in range(N):
        if classes[i] == classes[j]: 
            pairs.append([i, j])
pairs_tensor = torch.tensor(pairs).T

class LookUpGCN(torch.nn.Module):
    def __init__(self, num_phonemes, embed_dim, hidden_channels, out_channels):
        super(LookUpGCN, self).__init__()
        self.embedding = nn.Embedding(num_phonemes, embed_dim, padding_idx=num_phonemes-1)
        nn.init.xavier_uniform_(self.embedding.weight)     
        self.conv1 = GCNConv(embed_dim, hidden_channels, add_self_loops=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, add_self_loops=False)

    def forward(self, phoneme_indices, edge_index):
        x = self.embedding(phoneme_indices)
        residual = x 
        x = self.conv1(x, edge_index)
        x = residual + x
        residual = x
        x = self.conv2(x, edge_index)
        return residual + x
    
    def get_embedding(self, phoneme_indices):
        return self.embedding(phoneme_indices)


class GCN_MDD(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier_vocab = nn.Linear(1536, 41)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=768, num_heads=16, dropout=0.2, batch_first=True)
        self.look_up_model = LookUpGCN(41, 768, 768, 768)
        row = list(range(41))
        self.indices = torch.tensor(row,  dtype=torch.long).to('cuda')
        self.edge_index = pairs_tensor.to('cuda')
        self.linear = nn.Linear(768, 768)

        
    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()
    
    def look_up_table(self,canonical):
        return self.look_up_model(self.indices, self.edge_index)[canonical]

    def get_look_up_table(self):
        return self.look_up_model(self.indices, self.edge_index)

    def get_embedding(self):
        return self.look_up_model.get_embedding(self.indices)

    def forward(self, audio_input, canonical):
        acoustic = self.wav2vec2(audio_input, attention_mask=None)[0] #b x t x 768
        out = self.look_up_table(canonical) 
        Hk          = self.linear(out)
        Hv          = out
        o, _     = self.multihead_attention(acoustic, Hk, Hv)
        o        = torch.concat([acoustic, o], dim = 2)
        logits = self.classifier_vocab(o)
        return logits