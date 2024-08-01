from torch.utils.data import Dataset
import sentencepiece as spm
import torch
import string


class TextDataset(Dataset):
    def __init__(self, file_path, max_len, vocab_size):
        self.file_path = file_path
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sp_model_path = 'spm.model'
        self._prepare_data()

    def _prepare_data(self):
        self._train_tokenizer()
        self._load_tokenizer()
        self.tokens_num = self._tokenize_text()
        self.sentence, self.label = self._create_sequences(self.tokens_num)
        self.vocab_size = self._get_vocab_size()

    def _train_tokenizer(self):
        spm.SentencePieceTrainer.train(
            input=self.file_path, model_prefix='spm', vocab_size=self.vocab_size, 
            pad_id=0, unk_id=1, bos_id=2, eos_id=3, user_defined_symbols=['<pad>', '<bos>', '<eos>']
        )

    def _load_tokenizer(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.sp_model_path)

    def _tokenize_text(self):
        with open(self.file_path, "r", encoding="utf-8") as file:
            text = file.read().replace("\n", " ").lower()
        text = ''.join(char for char in text if char not in string.punctuation)
        tokens = self.sp.encode(text)
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

    def _get_vocab_size(self):
        return self.sp.get_piece_size()

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, idx):
        return self.sentence[idx], self.label[idx]

    def decode(self, token):
        token = token.cpu().numpy()
        return self.sp.decode(token.tolist())

