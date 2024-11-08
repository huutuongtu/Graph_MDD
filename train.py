
from jiwer import wer
import torch, json, os, librosa, transformers, gc
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor
import json
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
import torch.optim as optim
from torch.utils.data import DataLoader
from pyctcdecode import build_ctcdecoder
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import torch
from torch.utils.data import Dataset
import numpy as np
from dataloader import MDD_Dataset, collate_fn
import einops
from dataloader import text_to_tensor
from gcn_model import GCN_MDD, init_weights_xavier
from pyctcdecode import build_ctcdecoder
from jiwer import wer
import ast

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, padding_side='right',do_normalize=True, return_attention_mask=False)
min_wer = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epoch = 100

gc.collect()

df_train = pd.read_csv('./train_canonical_error.csv')
df_dev = pd.read_csv("./dev.csv")
train_dataset = MDD_Dataset(df_train)

batch_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

model = GCN_MDD.from_pretrained(
    'facebook/wav2vec2-base-100h' , 
)
def is_model_frozen(model):
    return all(param.requires_grad == True for param in model.parameters())


print("Not Frozen?", is_model_frozen(model))
model.freeze_feature_extractor()
model = model.to(device)
# model.look_up_model.apply(init_weights_xavier)


list_vocab = ['t ', 'uw ', 'er ', 'ah ', 'sh ', 'ng ', 'ow ', 'aw ', 'aa ', 'th ', 'ih ', 'zh ', 'k ', 'y ', 'l ', 'uh ', 'ch ', 'w ', 'b ', 'v ', 'ao ', 's ', 'p ', 'iy ', 'r ', 'eh ', 'f ', 'n ', 'ay ', 'oy ', 'd ', 'g ', 'ey ', 'err ', 'dh ', 'ae ', 'hh ', 'm ', 'jh ', 'z ', '']
decoder_ctc = build_ctcdecoder(
                              labels = list_vocab,
                              )

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
ctc_loss = nn.CTCLoss(blank = 40)
for epoch in range(num_epoch):
  model.train().to(device)
  running_loss = []
  print(f'EPOCH {epoch}:')
  for i, data in tqdm(enumerate(train_loader)):
    acoustic, linguistic, labels, error_gt, target_lengths  = data
    logits = model(acoustic, linguistic)
    logits = logits.transpose(0,1)
    input_lengths = torch.full(size=(logits.shape[1],), fill_value=logits.shape[0], dtype=torch.long, device=device)
    logits = F.log_softmax(logits, dim=2)
    loss = ctc_loss(logits, labels, input_lengths, target_lengths)

    running_loss.append(loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # break
  training_loss = sum(running_loss) / len(running_loss)

  print(f"Training loss: {training_loss}")
  if training_loss<=1: #ensure for fast ctc decode
    with torch.no_grad():
      model.eval().to(device)
      worderrorrate = []
      for point in tqdm(range(len(df_dev))):
        acoustic, _ = librosa.load("../EN_MDD/WAV/" + df_dev['Path'][point] + ".wav", sr=16000)
        acoustic = feature_extractor(acoustic, sampling_rate = 16000)
        acoustic = torch.tensor(acoustic.input_values, device=device)
        transcript = df_dev['Transcript'][point]
        canonical = df_dev['Canonical'][point]
        canonical = text_to_tensor(canonical)
        canonical = torch.tensor(canonical, dtype=torch.long, device=device)
        logits = model(acoustic, canonical.unsqueeze(0))
        logits = F.log_softmax(logits.squeeze(0), dim=1)
        x = logits.detach().cpu().numpy()
        hypothesis = decoder_ctc.decode(x).strip()
        # print(hypothesis)
        error = wer(transcript, hypothesis)
        worderrorrate.append(error)

      epoch_wer = sum(worderrorrate)/len(worderrorrate)
      if (epoch_wer < min_wer):
        print("save_checkpoint...")
        min_wer = epoch_wer
        torch.save(model.state_dict(), 'checkpoint/gcn_mdd.pth')
      print("wer checkpoint " + str(epoch) + ": " + str(epoch_wer))
      print("min_wer: " + str(min_wer))
      