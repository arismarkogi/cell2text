from transformers import PretrainedConfig


class Cell2TextConfig(PretrainedConfig):
    model_type = "cell2text"

    def __init__(
        self,
        gpt_config=None,
        cell_emb_mode="cell",
        cell_encoder_hidden_size=512,
        max_ncells=200,
        cell_emb_layer=-1,
        cell_emb_label=None,
        text_encoder_hidden_size=768,
        text_emb_layer=-1,
        forward_batch_size=-1,
        nproc=4,
        num_classes=0,
        embedding_fusion_method="attention",
        max_generation_length=100,
        num_beams=4,
        use_encoder_attention_mask=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.gpt_config = gpt_config or {}

        self.cell_emb_mode = cell_emb_mode
        self.cell_encoder_hidden_size = cell_encoder_hidden_size
        self.max_ncells = max_ncells
        self.cell_emb_layer = cell_emb_layer
        self.cell_emb_label = cell_emb_label or ["sample_name", "cell type rough", "cell type"]

        self.text_encoder_hidden_size = text_encoder_hidden_size
        self.text_emb_layer = text_emb_layer

        self.forward_batch_size = forward_batch_size
        self.nproc = nproc
        self.num_classes = num_classes

        self.embedding_fusion_method = embedding_fusion_method

        self.max_generation_length = max_generation_length
        self.num_beams = num_beams
        self.use_encoder_attention_mask = use_encoder_attention_mask
