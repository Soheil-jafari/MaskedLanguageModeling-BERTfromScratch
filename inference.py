# inference.py
import torch
from model import MaskedLanguageModel
from tokenizer import Tokenizer
from config import config


def predict_mask(text, model, tokenizer):
    model.eval()
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    if config.mask_token_id not in input_ids:
        print("No [MASK] token found in input text.")
        return text

    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(config.device)
    with torch.no_grad():
        logits = model(input_tensor)

    predicted_ids = logits.argmax(dim=-1).squeeze().tolist()
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids)

    reconstructed = []
    for orig, pred in zip(tokens, predicted_tokens):
        if orig == '[MASK]':
            reconstructed.append(pred)
        else:
            reconstructed.append(orig)

    return tokenizer.detokenize(reconstructed)


if __name__ == '__main__':
    tokenizer = Tokenizer()
    tokenizer.build_vocab(config.train_text_path)

    model = MaskedLanguageModel(config).to(config.device)
    checkpoint_path = f"{config.model_save_path}/model_epoch{config.num_epochs}.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))

    test_text = "The capital of France is [MASK]."
    output = predict_mask(test_text, model, tokenizer)
    print("Predicted Text:", output)
