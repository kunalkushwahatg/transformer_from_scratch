
import torch
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
SAVE_PATH = Config.SAVE_PATH

shape = (BATCH_SIZE, MAX_LEN, DMODEL)



dataset = TextDataset(DATA_FILE_PATH, max_len=MAX_LEN, vocab_size=VOCAB_SIZE)
print("Vocabulary size:", dataset.vocab_size)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)



# Initialize model, criterion, and optimizer
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

    average_loss = sum(losses) / len(losses)
    print(f'Epoch {epoch + 1}, Average Loss: {average_loss}')

#infrence 
TOKEN_GEN = 10000
text = "The quick brown fox jumps over the lazy dog"

def infrence(text):
    model.eval()
    for i in range(TOKEN_GEN):
        tokens = dataset.sp.encode(text)
        
        #acess last MAX_LEN tokens
        tokens = tokens[-MAX_LEN:]

        #convert to tensor
        tokens = torch.Tensor(tokens).long().unsqueeze(0).to(DEVICE)

        #get prediction
        prediction = model(tokens)
        prediction = prediction.squeeze(0)

        #get argmax
        prediction = torch.argmax(prediction,dim=-1)

        #decode 
        text += dataset.decode(prediction) + " "
    return text

print(infrence(text))
# Save the model
torch.save(model.state_dict(), SAVE_PATH)

print("Training completed.")

