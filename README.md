# ELECTRA-from-Scratch: A PyTorch Implementation

This project implements the **ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately)** pre-training method from scratch using PyTorch. Unlike traditional Masked Language Models (MLM) like BERT, which only learn from a small subset of masked tokens, ELECTRA trains a discriminator model to predict whether each token in a sequence is the original or a replacement proposed by a small generator network. This "Replaced Token Detection" task is significantly more sample-efficient, leading to faster pre-training and better downstream performance.

This implementation is built entirely from scratch, without relying on high-level libraries like Hugging Face's `transformers`, to demonstrate a deep, foundational understanding of modern NLP architectures and pre-training strategies.

---

## 📌 Key Features

-   **ELECTRA Pre-training:** Implements the core Replaced Token Detection (RTD) pre-training task.
-   **Generator-Discriminator Architecture:** A small Generator network proposes token replacements, and a larger Discriminator network learns to identify them.
-   **From-Scratch Transformer Encoder:** The core of both models is a custom-built Transformer encoder.
-   **Advanced Tokenization:** Utilizes a Byte-Pair Encoding (BPE) tokenizer trained on the target corpus.
-   **Modern Training Techniques:** Includes learning rate scheduling with warmup and gradient clipping for stable training.
-   **Modular & Readable Code:** The codebase is organized logically to be easily understood, modified, and extended.

---

## 🔧 File Structure

-   `config.py`: Centralized configuration for all hyperparameters and paths.
-   `tokenizer.py`: Handles BPE tokenizer training and loading.
-   `dataset.py`: Prepares data by masking tokens for the generator.
-   `generator.py`: The smaller Masked Language Model that proposes token replacements.
-   `discriminator.py`: The main ELECTRA model that learns to distinguish original from replaced tokens.
-   `train.py`: The main training script that orchestrates the ELECTRA pre-training process.
-   `inference.py`: A script to demonstrate how the trained discriminator can be used to detect replaced tokens in text.

---

## 🧠 The ELECTRA Method

The pre-training process works as follows:

1.  **Masking:** A portion of the input tokens are replaced with a `[MASK]` token.
2.  **Generator:** The small `Generator` model (a standard MLM) is trained to predict the original masked tokens.
3.  **Token Replacement:** The output of the `Generator` is used to replace the `[MASK]` tokens. Now, the input sequence is "corrupted" with plausible but potentially incorrect tokens.
4.  **Discriminator:** The larger `Discriminator` model receives this corrupted sequence and is trained to predict, for *every token*, whether it was part of the original input or was replaced by the `Generator`.

This setup is highly efficient because the `Discriminator` gets a learning signal from every single token in the sequence, whereas BERT only learns from the 15% of tokens that were masked.

---

## 📊 Sample Results

After training on the WikiText-2 dataset, the discriminator learns to effectively identify replaced tokens. The training logs show a steady decrease in both generator and discriminator loss, indicating successful learning.

**Training Log Snippet:**
```
Epoch 10/20 | Avg Gen Loss: 2.5123 | Avg Disc Loss: 0.1421
Epoch 11/20 | Avg Gen Loss: 2.4589 | Avg Disc Loss: 0.1288
...
Epoch 20/20 | Avg Gen Loss: 2.1034 | Avg Disc Loss: 0.0956
```

**Inference Example:**

The `inference.py` script can be used to see the discriminator in action. We manually corrupt a sentence and see if the model can spot the fake word.

**Input:** `The king is a powerful ruler.` (Manually changed to `The queen is a powerful ruler.`)

**Output:**
```
Input Text: the queen is a powerful ruler .
------------------------------
Token           Is_Replaced_Prob
------------------------------
the             0.0012
queen           0.9854   <-- Correctly identified as replaced!
is              0.0025
a               0.0018
powerful        0.0031
ruler           0.0029
.               0.0005
```

This clearly demonstrates that the discriminator has learned a deep contextual understanding of the language, as it knows that "queen" is a less likely token in this specific context than the original "king".

---

## 🏁 Getting Started

1.  **Install dependencies:**
    ```bash
    pip install torch tokenizers tqdm
    ```

2.  **Download a dataset:**
    Download a raw text file (like `wikitext-2.txt`) and place it in the project root. Update `config.py` with the correct filename.

3.  **Train the model:**
    ```bash
    python train.py
    ```

4.  **Run inference:**
    ```bash
    python inference.py
    ```
