# tokenizer.py
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from config import config
import os

def train_tokenizer(files):
    """
    Trains a BPE tokenizer on a list of files and saves it to disk.
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=config.vocab_size)
    
    tokenizer.train(files, trainer)
    
    # Save the tokenizer
    os.makedirs(os.path.dirname(config.tokenizer_path), exist_ok=True)
    tokenizer.save(config.tokenizer_path)

    return tokenizer

def get_tokenizer():
    """
    Loads a pre-trained tokenizer from disk, or trains a new one if it doesn't exist.
    """
    if os.path.exists(config.tokenizer_path):
        return Tokenizer.from_file(config.tokenizer_path)
    else:
        print("Training a new tokenizer...")
        with open(config.dataset_path, "r", encoding="utf-8") as f:
            # Create a temporary file with the text for the tokenizer to read
            temp_file = "temp_tokenizer_train_data.txt"
            with open(temp_file, "w", encoding="utf-8") as temp_f:
                for line in f:
                    temp_f.write(line)
        
        tokenizer = train_tokenizer([temp_file])
        os.remove(temp_file) # clean up temporary file
        return tokenizer
