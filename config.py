
import torch
class Config:

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # File paths
    DATA_FILE_PATH = "data/input.txt"

    # Model configuration
    DMODEL = 512
    HEADS = 4
    BATCH_SIZE = 32
    MAX_LEN = 10

    # Tokenizer configuration
    VOCAB_SIZE = 30000

    # Training configuration
    EPOCHS = 10
    LEARNING_RATE = 0.001
