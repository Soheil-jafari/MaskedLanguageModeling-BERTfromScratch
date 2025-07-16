# config.py

class Config:
    def __init__(self):
        # General Model Config
        self.vocab_size = 30522
        self.max_seq_len = 128
        self.hidden_dim = 256
        self.num_heads = 4
        self.num_layers = 4
        self.ffn_dim = 1024
        self.dropout = 0.1

        # Generator-specific Config (smaller than the discriminator)
        self.generator_hidden_dim = 64
        self.generator_num_layers = 2
        self.generator_num_heads = 2
        self.generator_ffn_dim = 256


        # Pre-training Config
        self.mask_prob = 0.15
        self.batch_size = 32
        self.num_epochs = 20 # Increased epochs for better convergence
        self.learning_rate = 5e-4
        self.warmup_steps = 1000
        self.max_grad_norm = 1.0

        # ELECTRA-specific Loss Weights
        self.generator_weight = 1.0
        self.discriminator_weight = 50.0

        # Environment & Paths
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.print_freq = 100
        self.model_save_path = "electra_model"
        self.tokenizer_path = "bpe_tokenizer.json"
        self.dataset_path = "wikitext-2.txt"

# Create a single instance of the config
config = Config()
