# BERT-Masked-Language-Model

This project implements a self-supervised BERT-style Masked Language Model (MLM) using PyTorch from scratch. It mirrors the core logic of original BERT pretraining without relying on HuggingFace Transformers.

---

## 📌 Features
- BERT-style transformer encoder
- Custom tokenizer and vocabulary builder
- 15% token masking strategy for training
- Trained using CrossEntropy loss with `ignore_index` on non-masked tokens
- Clean modular codebase (~500+ LOC total)
- Compatible with WikiText-2 and similar corpus

---

## 🔧 Files
- `tokenizer.py`: Simple word-level tokenizer and vocab builder
- `dataset.py`: Applies masking, returns (input_ids, labels) for MLM
- `model.py`: Transformer encoder with positional and token embeddings
- `train.py`: Training loop with epoch-based saving
- `inference.py`: Test the model with masked sentences
- `config.py`: All model and training settings

---

## 🧪 Sample Inference
```bash
$ python inference.py
Input: The capital of France is [MASK].
Output: The capital of France is Paris.
```

---

## 🧠 Results
Trained on a sample from WikiText-2 for 10 epochs. The model was able to reliably fill in masked tokens in syntactic and factual contexts after training, showing successful learning of contextual word representations.

---

## 📁 Example Dataset Structure
```
project_root/
├── train_text.txt  # plain text with one sentence per line
├── config.py
├── model.py
├── dataset.py
├── tokenizer.py
├── train.py
├── inference.py
```

---

## 🏁 Getting Started
```bash
pip install torch
python train.py
python inference.py
```

---

## ✅ Author
Built, tested, and run locally by Hiroki Kurano.
Fully custom and independently implemented for research and PhD-level demonstration.# MaskedLanguageModeling-BERTfromScratch
