from transformers import PretrainedConfig

class Cell2TextConfig(PretrainedConfig):
    model_type = "cell2text"
    
    def __init__(
        self,
        # Encoder configuration
        emb_mode="gene", # ["cls", "cell", "gene"], use "gene" for now and probably forever
        max_ncells=1000,
        emb_layer=-1, # {-1, 0}
        emb_label=None,
        nproc=4,
        forward_batch_size=100,
        summary_stat=None, # [None, "mean", "median", "exact_mean", "exact_median"]
        
        # Model dimensions
        cell_encoder_hidden_size=1152, # Geneformer hidden_dim
        
        
        projector = "mlp", # ["mlp", "perceiver"]
        
        # When useing the "mlp"
        mlp_hidden_size=2048, # projection layer hidden_dim
        mlp_dropout=0.05,
        top_k=512, # hyperparameter for the top_k selection of genes (tokens)

        # When using the "perceiver"
        num_latents = 128,
        preceiver_depth=4,
        num_heads=8,
        ff_mult=2,
        perceiver_dropout=0.1,
        use_position_encoding=True,
        perceiver_cross_attn_layers=1,
        perceiver_num_heads=4,



        
        # Paths and files
        token_dictionary_path="/home/arismarkog/Desktop/cell2text/Geneformer/geneformer/token_dictionary_gc95M.pkl",
        geneformer_path="/home/arismarkog/Desktop/cell2text/Geneformer",
        
        # Generation parameters - Fixed parameter names to match Cell2TextLlamaConfig
        max_new_tokens=500,  # Changed from max_length
        num_beams=1,
        early_stopping=True,
        no_repeat_ngram_size=2,
        temperature=1.0,
        top_p=1.0,
        
        # Model configurations
        llama_config=None, # Configuration for Llama model
        decoder_model_name_or_path="meta-llama/Llama-3.2-1B-Instruct", # Default model path
        decoder_hidden_size=2048, # Default for meta-llama/Llama-3.2-1B-Instruct
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Geneformer encoder config
        self.emb_mode = emb_mode
        self.max_ncells = max_ncells
        self.emb_layer = emb_layer
        self.emb_label = emb_label
        self.nproc = nproc
        self.forward_batch_size = forward_batch_size
        self.summary_stat = summary_stat
        
        # Model dimensions
        self.cell_encoder_hidden_size = cell_encoder_hidden_size
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_dropout = mlp_dropout
        self.top_k = top_k
        self.projector = projector


        self.num_latents = num_latents
        self.preceiver_depth = preceiver_depth
        self.num_heads = num_heads
        self.use_position_encoding=use_position_encoding
        self.perceiver_cross_attn_layers = perceiver_cross_attn_layers
        self.perceiver_num_heads = perceiver_num_heads
        self.ff_mult = ff_mult
        self.perceiver_dropout = perceiver_dropout
        
        # Paths
        self.token_dictionary_path = token_dictionary_path
        self.geneformer_path = geneformer_path
        
        # Generation parameters - Fixed attribute name
        self.max_new_tokens = max_new_tokens  # Changed from max_generation_length
        self.num_beams = num_beams
        self.early_stopping = early_stopping
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.temperature = temperature
        self.top_p = top_p
        
        # Model configurations
        self.llama_config = llama_config or {}
        self.decoder_model_name_or_path = decoder_model_name_or_path
        self.decoder_hidden_size = decoder_hidden_size