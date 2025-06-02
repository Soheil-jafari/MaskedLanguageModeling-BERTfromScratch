# tokenizer.py
import json
import re
from collections import Counter

class Tokenizer:
    def __init__(self, vocab_path=None):
        self.special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
        self.vocab = {}
        self.inv_vocab = {}
        if vocab_path:
            self.load_vocab(vocab_path)

    def build_vocab(self, texts, vocab_size=30522):
        word_freq = Counter()
        for line in texts:
            tokens = self.tokenize(line)
            word_freq.update(tokens)

        most_common = word_freq.most_common(vocab_size - len(self.special_tokens))
        self.vocab = {tok: i for i, tok in enumerate(self.special_tokens)}
        for word, _ in most_common:
            self.vocab[word] = len(self.vocab)

        self.inv_vocab = {i: t for t, i in self.vocab.items()}

    def save_vocab(self, path):
        with open(path, 'w') as f:
            json.dump(self.vocab, f)

    def load_vocab(self, path):
        with open(path, 'r') as f:
            self.vocab = json.load(f)
        self.inv_vocab = {int(i): t for t, i in self.vocab.items()}

    def tokenize(self, text):
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def encode(self, text):
        tokens = ["[CLS]"] + self.tokenize(text)[:126] + ["[SEP]"]
        token_ids = [self.vocab.get(tok, self.vocab["[UNK]"]) for tok in tokens]
        padding = [self.vocab["[PAD]"]] * (128 - len(token_ids))
        return token_ids + padding

    def decode(self, token_ids):
        tokens = [self.inv_vocab.get(i, "[UNK]") for i in token_ids]
        return " ".join(tokens).replace(" [PAD]", "").strip()
