# train.py

import torch
import random
import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from model.PretrainingEncoder import Pretraining
from dataset import TextDataset
from config import Config

# Load configurations
DEVICE = Config.DEVICE
DATA_FILE_PATH = Config.DATA_FILE_PATH
DMODEL = Config.DMODEL
HEADS = Config.HEADS
BATCH_SIZE = Config.BATCH_SIZE
MAX_LEN = Config.MAX_LEN
VOCAB_SIZE = Config.VOCAB_SIZE
EPOCHS = Config.EPOCHS
LEARNING_RATE = Config.LEARNING_RATE

# Load dataset
dataset = TextDataset("data/input.txt", max_len=10, vocab_size=10000)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


# Model shape
shape = (BATCH_SIZE, MAX_LEN, DMODEL)

# Initialize model
model = Pretraining(VOCAB_SIZE, shape,DEVICE ,heads=HEADS).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    losses = []
    running_loss = 0.0
    model.train()

    for b in tqdm.tqdm(dataloader, desc=f'Epoch {epoch + 1}/{EPOCHS - 1}'):
        inputs, targets = b[0].to(DEVICE), b[1].to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        losses.append(loss.item())

    average_loss = running_loss / len(batches)
    print(f'Epoch {epoch + 1}, Average Loss: {average_loss}')

print("Training completed.")
