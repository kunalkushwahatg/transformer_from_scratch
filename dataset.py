import torch
import string
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, file_path, max_len=10, vocab_size=30000):
        self.file_path = file_path
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = ByteLevelBPETokenizer()
        self._prepare_data()

    def _prepare_data(self):
        self._train_tokenizer()
        self._load_tokenizer()
        self.tokens_num = self._tokenize_text()
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.sentence, self.label = self._create_sequences(self.tokens_num)

    def _train_tokenizer(self):
        self.tokenizer.train(files=[self.file_path], vocab_size=self.vocab_size, min_frequency=2)

    def _load_tokenizer(self):
        self.tokenizer.save_model(".", "bpe")
        self.tokenizer = ByteLevelBPETokenizer(
            "./bpe-vocab.json",
            "./bpe-merges.txt",
        )

    def _tokenize_text(self):
        with open(self.file_path, "r", encoding="utf-8") as file:
            text = file.read().replace("\n", " ").lower()
        text = ''.join(char for char in text if char not in string.punctuation)
        tokens = self.tokenizer.encode(text).ids
        return tokens

    def _create_sequences(self, tokens_num):
        x = []
        y = []
        for i in range(len(tokens_num) - self.max_len - 1):
            x.append(tokens_num[i:self.max_len + i])
            y.append(tokens_num[self.max_len + i])
        sentence = torch.Tensor(x).long()
        label = torch.Tensor(y).long()
        return sentence, label

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, idx):
        return self.sentence[idx], self.label[idx]

    def decode(self, token):
        token = token.cpu().numpy()
        return self.tokenizer.decode(token)

if __name__ == "__main__":
    dataset = TextDataset("data/input.txt", max_len=10, vocab_size=10000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"Vocab size: {dataset.vocab_size}")

    for batch_idx, (sentences, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx+1}:")
        print("Sentences:", sentences)
        print("Labels:", labels)
        print("First sentence in the batch:", dataset.decode(sentences[0]))
        if batch_idx == 0:
            break 
