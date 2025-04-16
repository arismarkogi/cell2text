import torch
import logging
from transformers import BertForMaskedLM, BertConfig
from typing import Optional, Union, Tuple
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.configuration_utils import PretrainedConfig
#from Geneformer.geneformer.in_silico_perturber import quant_layers
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Set as constants here, so they are available in the TranscriptomeProcessor
PAD_TOKEN_ID = 0
MODEL_INPUT_SIZE = 2048


class GeneformerConfig(PretrainedConfig):
    model_type = "geneformer"

    def __init__(
        self,
        num_classes=0,
        emb_mode="cell",
        hidden_size=512,
        max_ncells=200,
        emb_layer=-1,
        emb_label=["sample_name", "cell type rough", "cell type"],
        forward_batch_size=-1,
        nproc=4,
        summary_stat=None,
        vocab_size=25426,  # genes+2 for <mask> and <pad> tokens
        num_hidden_layers=12,
        initializer_range=0.2,
        layer_norm_eps=1e-12,
        attention_probs_dropout_prob=0.02,
        hidden_dropout_prob=0.02,
        intermediate_size=1024,
        hidden_act="relu",
        max_position_embeddings=2**11,
        num_attention_heads=4,
        output_hidden_states=True,
        output_attentions=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_type = GeneformerConfig.model_type
        self.num_classes = num_classes
        self.emb_mode = emb_mode
        self.hidden_size = hidden_size
        self.max_ncells = max_ncells
        self.emb_layer = emb_layer
        self.emb_label = emb_label
        self.forward_batch_size = forward_batch_size
        self.nproc = nproc
        self.summary_stat = summary_stat
        
        # BERT-specific configuration
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.pad_token_id = PAD_TOKEN_ID
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions


class GeneformerModel(PreTrainedModel):
    config_class = GeneformerConfig
    base_model_prefix = "geneformer_model"
    is_parallelizable = False
    main_input_name = "expression_tokens"

    def __init__(self, config: GeneformerConfig):
        super().__init__(config)
        self.config = config
        
        # Validate emb_mode
        valid_emb_modes = ["cell", "gene", "cls", "mean"]
        if config.emb_mode not in valid_emb_modes:
            raise ValueError(f"Invalid emb_mode: {config.emb_mode}, must be one of {valid_emb_modes}")

        # Extract bert-specific configuration
        bert_config = BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            initializer_range=config.initializer_range,
            layer_norm_eps=config.layer_norm_eps,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            hidden_dropout_prob=config.hidden_dropout_prob,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            max_position_embeddings=config.max_position_embeddings,
            model_type="bert",
            num_attention_heads=config.num_attention_heads,
            pad_token_id=PAD_TOKEN_ID,
            output_hidden_states=config.output_hidden_states,
            output_attentions=config.output_attentions,
        )

        self.geneformer_model = BertForMaskedLM(bert_config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        expression_tokens: torch.Tensor,
        expression_token_lengths: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        Process gene expression data through the model to generate embeddings.
        """
        
        # Fallback implementation if get_embs fails or is not available
        # Create attention mask from token lengths
        batch_size, seq_len = expression_tokens.shape
        attention_mask = torch.zeros(batch_size, seq_len, device=expression_tokens.device, dtype=torch.bool)
        
        for i, length in enumerate(expression_token_lengths):
            attention_mask[i, :length] = True
        
        # Run BERT model
        outputs = self.geneformer_model(
            input_ids=expression_tokens,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get hidden states from the specified layer
        hidden_states = outputs.hidden_states
        layer_idx = getattr(self.config, 'emb_layer', -1)
        if layer_idx < 0:
            layer_idx = len(hidden_states) + layer_idx
        
        target_hidden_state = hidden_states[layer_idx]
        
        # Get embeddings based on emb_mode
        emb_mode = getattr(self.config, 'emb_mode', 'cls')
        if emb_mode == "cls":
            # Get CLS token embedding (first token)
            embs = target_hidden_state[:, 0]
        elif emb_mode == "mean":
            # Mean of non-padding tokens
            non_padding_mask = (expression_tokens != PAD_TOKEN_ID).float().unsqueeze(-1)
            sum_embeddings = torch.sum(target_hidden_state * non_padding_mask, dim=1)
            token_counts = torch.sum(non_padding_mask, dim=1)
            embs = sum_embeddings / (token_counts + 1e-10)  # Add small epsilon to avoid division by zero
        elif emb_mode == "cell":
            # For cell mode, use the CLS token embedding as in the original Geneformer
            embs = target_hidden_state[:, 0]
        elif emb_mode == "gene":
            # For gene mode, use the entire hidden state
            embs = target_hidden_state
        else:
            raise ValueError(f"Unsupported emb_mode: {emb_mode}")
        
        return (outputs, embs)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Optional[str] = None, *model_args, **kwargs
    ) -> PreTrainedModel:
        """
        Load a pretrained Geneformer model from a local directory or a custom path.
        
        Args:
            pretrained_model_name_or_path: Path to local directory containing model weights
                or auto-detect from environment
            *model_args: Additional positional arguments to pass to the model
            **kwargs: Additional keyword arguments to pass to the model
            
        Returns:
            A pretrained GeneformerModel instance
        """
        # Handle case where no path is provided
        if pretrained_model_name_or_path is None:
            # First check if GENEFORMER_PATH environment variable is set
            env_path = os.environ.get("GENEFORMER_PATH")
            if env_path and os.path.exists(env_path):
                pretrained_model_name_or_path = env_path
                logger.info(f"Using Geneformer path from environment: {env_path}")
            else:
                # Try to find Geneformer in the installed packages
                try:
                    import importlib.metadata
                    import site
                    import sys
                    
                    # Check if Geneformer is installed
                    try:
                        geneformer_dist = importlib.metadata.distribution("geneformer")
                        # Look for model files in the package directory
                        for site_path in site.getsitepackages() + [site.getusersitepackages()]:
                            potential_path = os.path.join(site_path, "geneformer", "pretrained")
                            if os.path.exists(potential_path):
                                pretrained_model_name_or_path = potential_path
                                logger.info(f"Found Geneformer pretrained model at: {potential_path}")
                                break
                    except importlib.metadata.PackageNotFoundError:
                        pass
                    
                    # If still not found, use the current directory as a last resort
                    if pretrained_model_name_or_path is None:
                        current_dir = os.getcwd()
                        if os.path.exists(os.path.join(current_dir, "Geneformer")):
                            pretrained_model_name_or_path = os.path.join(current_dir, "Geneformer")
                            logger.info(f"Using Geneformer directory in current path: {pretrained_model_name_or_path}")
                except Exception as e:
                    logger.warning(f"Error finding Geneformer installation: {e}")
            
            # If still not found, use the default HuggingFace repo as a fallback
            if pretrained_model_name_or_path is None:
                pretrained_model_name_or_path = "ctheodoris/Geneformer"
                logger.warning(
                    "No Geneformer path found. Using the Hugging Face model hub as fallback. "
                    "This may not work as expected. Consider installing Geneformer: "
                    "git clone https://huggingface.co/ctheodoris/Geneformer && cd Geneformer && pip install ."
                )
        
        # Extract configuration parameters from kwargs
        config_params = {}
        config_keys = [
            "num_classes", "emb_mode", "hidden_size", "max_ncells", 
            "emb_layer", "emb_label", "forward_batch_size", "nproc", "summary_stat"
        ]
        
        for key in config_keys:
            if key in kwargs:
                config_params[key] = kwargs.pop(key)
        
        # Check if config is provided directly
        if "config" in kwargs:
            config = kwargs.pop("config")
            if isinstance(config, dict):
                config = GeneformerConfig(**config)
            elif not isinstance(config, GeneformerConfig):
                raise ValueError(
                    "Parameter `config` must be a dictionary or an instance of `GeneformerConfig`."
                )
        else:
            # Try to load config from the model path
            try:
                config_path = os.path.join(pretrained_model_name_or_path, "config.json")
                if os.path.exists(config_path):
                    config = GeneformerConfig.from_pretrained(config_path)
                    # Update with any configuration parameters from kwargs
                    for key, value in config_params.items():
                        setattr(config, key, value)
                else:
                    # Create a default config
                    config = GeneformerConfig(**config_params)
            except Exception as e:
                logger.warning(f"Error loading configuration from {pretrained_model_name_or_path}: {e}")
                config = GeneformerConfig(**config_params)
        
        # Initialize model with config
        model = cls(config)
        
        # Check if the path exists and looks like a model directory
        if os.path.isdir(pretrained_model_name_or_path):
            # Check common model file patterns
            potential_model_files = [
                os.path.join(pretrained_model_name_or_path, "pytorch_model.bin"),
                os.path.join(pretrained_model_name_or_path, "model.safetensors"),
                # Look for model in subdirectories too
                os.path.join(pretrained_model_name_or_path, "pretrained", "pytorch_model.bin"),
                os.path.join(pretrained_model_name_or_path, "pretrained", "model.safetensors"),
            ]
            
            model_file = None
            for file_path in potential_model_files:
                if os.path.exists(file_path):
                    model_file = os.path.dirname(file_path)
                    break
                    
            if model_file:
                logger.info(f"Found model file at: {model_file}")
                pretrained_model_name_or_path = model_file
        
        # Load pretrained weights
        try:
            # First try to load as BertForMaskedLM which is the expected structure
            from transformers import BertConfig

            # Create a BERT config using your custom Geneformer config
            bert_config = BertConfig(
                vocab_size=20275,
                max_position_embeddings=4096,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                initializer_range=config.initializer_range,
                layer_norm_eps=config.layer_norm_eps,
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                hidden_dropout_prob=config.hidden_dropout_prob,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                num_attention_heads=config.num_attention_heads,
                pad_token_id=config.pad_token_id,
                output_hidden_states=config.output_hidden_states,
                output_attentions=config.output_attentions,
            )

            model.geneformer_model = BertForMaskedLM.from_pretrained(
                pretrained_model_name_or_path,
                config=bert_config,
                **kwargs
            )

            logger.info(f"Successfully loaded Geneformer model from {pretrained_model_name_or_path}")
        except Exception as e:
            logger.warning(f"Error loading model from {pretrained_model_name_or_path}: {e}")
            logger.warning("Model weights could not be loaded. Using initialized model.")
        
        return model