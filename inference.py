# inference.py
import torch
from discriminator import Discriminator
from tokenizer import get_tokenizer
from config import config

def predict_replaced(text, discriminator, tokenizer):
    discriminator.eval()
    
    # Manually corrupt the text for demonstration
    tokens = tokenizer.encode(text).tokens
    token_ids = tokenizer.encode(text).ids
    
    # Replace a token with a plausible but incorrect alternative
    if "king" in tokens:
        king_idx = tokens.index("king")
        tokens[king_idx] = "queen"
        token_ids = tokenizer.encode(" ".join(tokens)).ids

    input_tensor = torch.tensor([token_ids], dtype=torch.long).to(config.device)
    
    with torch.no_grad():
        logits = discriminator(input_tensor)
        probs = torch.sigmoid(logits).squeeze()

    print("Input Text:", " ".join(tokens))
    print("-" * 30)
    print("Token\t\tIs_Replaced_Prob")
    print("-" * 30)
    for token, prob in zip(tokens, probs):
        print(f"{token:<15}\t{prob.item():.4f}")

if __name__ == '__main__':
    tokenizer = get_tokenizer()
    
    discriminator = Discriminator().to(config.device)
    
    # Load the trained discriminator checkpoint
    # Make sure to change the epoch number to the one you want to use
    checkpoint_path = os.path.join(config.model_save_path, f"discriminator_epoch_{config.num_epochs}.pt")
    if os.path.exists(checkpoint_path):
        discriminator.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
        print("Loaded discriminator from checkpoint.")
    else:
        print("Could not find a trained discriminator checkpoint. Please train the model first.")
        exit()

    test_text = "The king is a powerful ruler."
    predict_replaced(test_text, discriminator, tokenizer)
