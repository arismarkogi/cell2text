import torch
import torch.nn as nn
from einops import repeat
import math
from transformers import BertTokenizer, BertModel
from .blip2_base import Blip2Base  # Assuming you have the blip2_base module available


class MLPProjectionLayer(nn.Module):
    """
    Creates a multi-layer perceptron (MLP) projection layer with GELU activation.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_prob: float = 0.0, bias: bool = True) -> None:
        super().__init__()

        layers = [
            nn.Linear(input_dim, hidden_dim, bias=bias),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        ]
        if dropout_prob > 0:
            layers.append(nn.Dropout(p=dropout_prob))
        layers.extend([
            nn.Linear(hidden_dim, output_dim, bias=bias),
            nn.LayerNorm(output_dim)
        ])
        self.projection = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class QFormerProjector(Blip2Base):
    """
    QFormer projector with BioBERT-Large, trainable query tokens, 
    and trainable last few BERT layers
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_query_tokens: int = 256,
        cross_attention_freq: int = 2,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_query_tokens = num_query_tokens
        
        # Hard-coded BioBERT-Large
        #bert_model_name = 'dmis-lab/biobert-base-cased-v1.2'
        bert_hidden_size = 768  # BioBERT-Large hidden size
        

        # Input projection to BioBERT-Large dimension
        self.input_projection = nn.Linear(input_dim, bert_hidden_size)
        
        # Initialize QFormer with BioBERT-Large
        self.Qformer, self.query_tokens = self.init_Qformer(
            model_name="pubmedbert",
            num_query_token=num_query_tokens,
            graph_width=bert_hidden_size,
            cross_attention_freq=cross_attention_freq, #cross_attention_freq,
            use_flash_attn=use_flash_attn
        )
        
        # Output projection
        self.output_projection = nn.Linear(bert_hidden_size, output_dim)
        
        # Layer normalization
        self.ln_cell = nn.LayerNorm(bert_hidden_size)
        
        # Configure trainable parameters
        self._configure_trainable_params()
        
        print(f"✓ BioBERT-Large QFormer initialized")
        print(f"✓ Query tokens: {num_query_tokens} (trainable)")
        self._print_trainable_summary()
    
    def _configure_trainable_params(self):
        """Configure trainable parameters: query tokens, cross-attention, and query-specific FFNs"""
        
        # 1. Freeze all QFormer parameters initially
        for param in self.Qformer.parameters():
            param.requires_grad = True
        
        # 2. Make query tokens trainable
        self.query_tokens.requires_grad = True
        print("✅ Query tokens: TRAINABLE")
        
        # 3. Make ALL cross-attention components trainable (all layers)
        cross_attn_count = 0
        for i, layer in enumerate(self.Qformer.bert.encoder.layer):
            if hasattr(layer, 'crossattention'):
                # Cross-attention components
                for param in layer.crossattention.parameters():
                    param.requires_grad = True
                cross_attn_count += 1
                print(f"✅ Layer {i}: cross-attention TRAINABLE")
        
        # 4. Make ALL query-specific FFN components trainable (all layers)
        query_ffn_count = 0
        for i, layer in enumerate(self.Qformer.bert.encoder.layer):
            # Query-specific intermediate layer
            if hasattr(layer, 'intermediate_query'):
                for param in layer.intermediate_query.parameters():
                    param.requires_grad = True
                query_ffn_count += 1
                print(f"✅ Layer {i}: intermediate_query TRAINABLE")
            
            # Query-specific output layer  
            if hasattr(layer, 'output_query'):
                for param in layer.output_query.parameters():
                    param.requires_grad = True
                print(f"✅ Layer {i}: output_query TRAINABLE")
        
        # 5. Keep projection layers trainable
        for param in self.input_projection.parameters():
            param.requires_grad = True
        for param in self.output_projection.parameters():
            param.requires_grad = True
        for param in self.ln_cell.parameters():
            param.requires_grad = True
        print("✅ Projection layers: TRAINABLE")
        
        print(f"\n📊 Summary:")
        print(f"   • Cross-attention layers: {cross_attn_count}")
        print(f"   • Query FFN layers: {query_ffn_count}")
        print(f"   • All randomly-initialized components are now trainable!")

    def _print_trainable_summary(self):
        """Print detailed summary of trainable parameters"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"\n📈 PARAMETER SUMMARY:")
        print(f"Trainable: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.1f}%)")
        
        # Component breakdown
        query_params = self.query_tokens.numel()
        
        # Projection layers
        proj_params = (
            sum(p.numel() for p in self.input_projection.parameters()) +
            sum(p.numel() for p in self.output_projection.parameters()) +
            sum(p.numel() for p in self.ln_cell.parameters())
        )
        
        # Count cross-attention and query FFN parameters
        cross_attn_params = 0
        query_ffn_params = 0
        
        for layer in self.Qformer.bert.encoder.layer:
            # Cross-attention parameters
            if hasattr(layer, 'crossattention'):
                cross_attn_params += sum(p.numel() for p in layer.crossattention.parameters() if p.requires_grad)
            
            # Query FFN parameters
            if hasattr(layer, 'intermediate_query'):
                query_ffn_params += sum(p.numel() for p in layer.intermediate_query.parameters() if p.requires_grad)
            if hasattr(layer, 'output_query'):
                query_ffn_params += sum(p.numel() for p in layer.output_query.parameters() if p.requires_grad)
        
        print(f"\n🔍 BREAKDOWN:")
        print(f"   • Query tokens: {query_params:,}")
        print(f"   • Cross-attention (all layers): {cross_attn_params:,}")
        print(f"   • Query FFNs (all layers): {query_ffn_params:,}")
        print(f"   • Projection layers: {proj_params:,}")
        print(f"   • Total trainable: {query_params + cross_attn_params + query_ffn_params + proj_params:,}")
        
        print(f"\n✅ STRATEGY: Train only randomly-initialized components")
        print(f"✅ FROZEN: All pre-trained BERT components (self-attention, standard FFNs)")

    def forward(self, cell_embeddings: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass - what actually happens during cross-attention
        """
        batch_size = cell_embeddings.size(0)
        device = cell_embeddings.device
        
        # Project input to BioBERT dimension
        projected_embeddings = self.input_projection(cell_embeddings)
        projected_embeddings = self.ln_cell(projected_embeddings)
        
        # Prepare query tokens [batch_size, num_query_tokens, hidden_size]
        query_tokens = self.query_tokens.expand(batch_size, -1, -1).to(device)
        
        if attention_mask is None:
            attention_mask = torch.ones(
                cell_embeddings.size()[:2], dtype=torch.long, device=device
            )
        
        # During QFormer.bert() forward pass:
        # 1. Query tokens go through self-attention (attend to each other)
        # 2. Every cross_attention_freq layers: query tokens attend to projected_embeddings
        # 3. This cross-attention is WHERE THE MAGIC HAPPENS - queries learn to extract
        #    relevant information from your cell embeddings
        # 4. Feed-forward networks process the attended representations
        
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,                    # [B, num_queries, H] 
            encoder_hidden_states=projected_embeddings,   # [B, seq_len, H] - your cell data
            encoder_attention_mask=attention_mask,        # [B, seq_len]
            use_cache=False,
            return_dict=True,
        )
        
        # Extract final query representations
        query_embeddings = query_output.last_hidden_state[:, :self.num_query_tokens, :]
        output_embeddings = self.output_projection(query_embeddings)
        
        return output_embeddings  # [batch_size, num_query_tokens, output_dim]
            
class PerceiverIO(nn.Module):
    """
    Perceiver IO implementation with both cross-attention and self-attention
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_latents: int = 128,
        num_cross_attn_layers: int = 1,  
        num_self_attn_layers: int = 4,   
        num_heads: int = 8,
        ff_mult: int = 2,
        dropout: float = 0.1,
        use_position_encoding: bool = True,
        max_seq_len: int = 10000,
    ):
        """
        Args:
            input_dim: Dimension of input embeddings
            output_dim: Dimension of output embeddings and latents
            num_latents: Number of learnable latent queries
            num_cross_attn_layers: Number of cross-attention layers (usually 1)
            num_self_attn_layers: Number of self-attention layers (the main processing)
            num_heads: Number of attention heads
            ff_mult: Feedforward dimension multiplier
            dropout: Dropout rate
            use_position_encoding: Whether to add positional encodings to inputs
            max_seq_len: Maximum sequence length for position encodings
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_latents = num_latents
        self.num_cross_attn_layers = num_cross_attn_layers
        self.num_self_attn_layers = num_self_attn_layers
        self.use_position_encoding = use_position_encoding
        
        # Learnable latent queries (the "bottleneck")
        self.latents = nn.Parameter(torch.randn(num_latents, output_dim))
        
        # Input projection to match latent dimension
        self.input_projection = nn.Linear(input_dim, output_dim)
        
        # Position encoding for inputs (optional but often helpful)
        if use_position_encoding:
            self.position_encoding = PositionalEncoding(output_dim, max_seq_len)
        
        # Cross-attention layers (encode inputs into latents)
        self.cross_attention_layers = nn.ModuleList([
            PerceiverCrossAttentionLayer(
                dim=output_dim,
                num_heads=num_heads,
                ff_mult=ff_mult,
                dropout=dropout
            ) for _ in range(num_cross_attn_layers)
        ])
        
        # Self-attention layers (process latents - this is the key Perceiver innovation)
        self.self_attention_layers = nn.ModuleList([
            PerceiverSelfAttentionLayer(
                dim=output_dim,
                num_heads=num_heads,
                ff_mult=ff_mult,
                dropout=dropout
            ) for _ in range(num_self_attn_layers)
        ])
        
        # Final output normalization
        self.final_norm = nn.LayerNorm(output_dim)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters"""
        nn.init.normal_(self.latents, mean=0.0, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass through Perceiver
        
        Args:
            inputs: [batch_size, seq_len, input_dim] - input embeddings
            mask: [batch_size, seq_len] - attention mask (True = valid, False = padding)
        
        Returns:
            [batch_size, num_latents, output_dim] - processed latent representation
        """
        batch_size = inputs.size(0)
        
        # Project input to output dimension
        projected_inputs = self.input_projection(inputs)
        
        # Add position encoding if enabled
        if self.use_position_encoding:
            projected_inputs = self.position_encoding(projected_inputs)
        
        # Initialize latents for this batch
        latents = repeat(self.latents, 'q d -> b q d', b=batch_size)
        
        # Phase 1: Cross-attention (encode inputs into latents)
        for cross_layer in self.cross_attention_layers:
            latents = cross_layer(latents, projected_inputs, mask)
        
        # Phase 2: Self-attention (process latents - the main computation)
        for self_layer in self.self_attention_layers:
            latents = self_layer(latents)
        
        # Final normalization
        output = self.final_norm(latents)
        
        return output


class PerceiverCrossAttentionLayer(nn.Module):
    """
    Cross-attention layer: latents attend to inputs
    """
    
    def __init__(self, dim: int, num_heads: int = 8, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        # Cross-attention: latents (queries) attend to inputs (keys, values)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(dim, int(dim * ff_mult)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * ff_mult), dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms (pre-norm style)
        self.norm1 = nn.LayerNorm(dim)  # Before cross-attention
        self.norm2 = nn.LayerNorm(dim)  # Before feedforward
        
    def forward(self, latents: torch.Tensor, inputs: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            latents: [batch_size, num_latents, dim] - latent queries
            inputs: [batch_size, seq_len, dim] - input sequence to attend to
            mask: [batch_size, seq_len] - attention mask (True = valid, False = padding)
        
        Returns:
            [batch_size, num_latents, dim] - updated latents
        """
        # Cross-attention with residual connection
        norm_latents = self.norm1(latents)
        
        # Convert mask format if provided
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # Invert for PyTorch convention
        
        attn_out, _ = self.cross_attn(
            query=norm_latents,
            key=inputs,
            value=inputs,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        latents = latents + attn_out
        
        # Feedforward with residual connection
        ff_out = self.feedforward(self.norm2(latents))
        latents = latents + ff_out
        
        return latents


class PerceiverSelfAttentionLayer(nn.Module):
    """
    Self-attention layer: latents attend to other latents
    This is the key innovation of the Perceiver - processing in the latent space
    """
    
    def __init__(self, dim: int, num_heads: int = 8, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        # Self-attention: latents attend to latents
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(dim, int(dim * ff_mult)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * ff_mult), dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms (pre-norm style)
        self.norm1 = nn.LayerNorm(dim)  # Before self-attention
        self.norm2 = nn.LayerNorm(dim)  # Before feedforward
        
    def forward(self, latents: torch.Tensor):
        """
        Args:
            latents: [batch_size, num_latents, dim] - latent queries
        
        Returns:
            [batch_size, num_latents, dim] - updated latents
        """
        # Self-attention with residual connection
        norm_latents = self.norm1(latents)
        
        attn_out, _ = self.self_attn(
            query=norm_latents,
            key=norm_latents,
            value=norm_latents,
            need_weights=False
        )
        
        latents = latents + attn_out
        
        # Feedforward with residual connection
        ff_out = self.feedforward(self.norm2(latents))
        latents = latents + ff_out
        
        return latents


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding
    """
    
    def __init__(self, dim: int, max_len: int = 10000):
        super().__init__()
        
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, dim]
        Returns:
            [batch_size, seq_len, dim] with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]