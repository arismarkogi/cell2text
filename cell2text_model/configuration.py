from transformers import PretrainedConfig


class Cell2TextConfig(PretrainedConfig):
    model_type = "cell2text"

    def __init__(
        self,
        gpt_config=None,
        emb_mode="cell", # ["cls", "cell", "gene"], this is a hyperparameter of the embeddings, "cell" is recommended
        max_ncells=1000,
        emb_layer=-1, #{-1, 0}
        emb_label=None,
        nproc = 4,
        forward_batch_size = 100,
        summary_stat = None, # [None, "mean", "median", "exact_mean", "exact_median"],
        cell_encoder_hidden_size = 512,
        token_dictionary_path = "/home/arismarkog/Desktop/cell2text/Geneformer/geneformer/token_dictionary_gc95M.pkl",
        geneformer_path = "/home/arismarkog/Desktop/cell2text/Geneformer",
        max_generation_length=100,
        num_beams=4,
        use_encoder_attention_mask=True,
        **kwargs,
    ):
        super().__init__(**kwargs)


        self.emb_mode = emb_mode
        self.max_ncells = max_ncells
        self.emb_layer = emb_layer
        self.emb_label = emb_label
        self.nproc = nproc
        self.forward_batch_size = forward_batch_size
        self.summary_stat = summary_stat

        self.cell_encoder_hidden_size = cell_encoder_hidden_size
        self.token_dictionary_path = token_dictionary_path
        self.geneformer_path = geneformer_path

        self.gpt_config = gpt_config or {}

        self.max_generation_length = max_generation_length
        self.num_beams = num_beams
        self.use_encoder_attention_mask = use_encoder_attention_mask
