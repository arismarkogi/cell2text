import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import logging
from typing import Dict, List, Tuple, Union, Optional

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate
        num_warmup_steps: The number of steps for the warmup phase
        num_training_steps: The total number of training steps
        last_epoch: The index of the last epoch when resuming training
        
    Returns:
        torch.optim.lr_scheduler: The learning rate scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
        
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for multimodal learning (NT-Xent/InfoNCE loss)
    
    Implements the NT-Xent loss from the paper:
    "A Simple Framework for Contrastive Learning of Visual Representations"
    """
    def __init__(self, temperature=0.07, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        
    def forward(self, embeddings_a, embeddings_b, labels=None):
        """
        Compute NT-Xent loss between embeddings from two different modalities
        
        Args:
            embeddings_a: Tensor of shape [batch_size, feature_dim] for the first modality
            embeddings_b: Tensor of shape [batch_size, feature_dim] for the second modality
            labels: Optional tensor of shape [batch_size] indicating positive pairs
                   If None, assumes diagonal entries (i,i) are positive pairs
                   
        Returns:
            loss: Scalar contrastive loss
        """
        batch_size = embeddings_a.shape[0]
        
        # Normalize the embeddings along the feature dimension
        embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=1)
        
        # Calculate the similarity matrix
        # Shape: [batch_size, batch_size]
        logits = torch.matmul(embeddings_a, embeddings_b.t()) / self.temperature
        
        # If no labels are provided, assume the diagonal are positive pairs
        if labels is None:
            labels = torch.arange(batch_size, device=logits.device)
            
        # Calculate the loss using cross-entropy
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        
        return loss
    
    def bidirectional_contrastive_loss(self, embeddings_a, embeddings_b):
        """
        Compute bi-directional contrastive loss: a->b and b->a
        
        Args:
            embeddings_a: Tensor of shape [batch_size, feature_dim] for the first modality
            embeddings_b: Tensor of shape [batch_size, feature_dim] for the second modality
            
        Returns:
            loss: Scalar bi-directional contrastive loss
        """
        # Forward direction: a -> b
        forward_loss = self.forward(embeddings_a, embeddings_b)
        
        # Backward direction: b -> a
        backward_loss = self.forward(embeddings_b, embeddings_a)
        
        # Average the losses
        total_loss = (forward_loss + backward_loss) / 2
        
        return total_loss



def save_checkpoint(model, optimizer, scheduler, epoch, global_step, args, best=False):
    """
    Save a checkpoint of the training
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        scheduler: The learning rate scheduler to save
        epoch: Current epoch number
        global_step: Global step number
        args: Training arguments
        best: Whether this is the best model so far
    """
    # Create checkpoint
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
    }
    
    # Determine checkpoint path
    if best:
        output_path = os.path.join(args.output_dir, "best_model.pt")
    else:
        output_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
    
    # Save checkpoint
    torch.save(checkpoint, output_path)
    logger.info(f"Saved checkpoint to {output_path}")

def load_checkpoint(model, optimizer=None, scheduler=None, checkpoint_path=None):
    """
    Load a checkpoint into the model and optionally optimizer and scheduler
    
    Args:
        model: The model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        epoch: Last completed epoch
        global_step: Global step
    """
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint {checkpoint_path} not found")
        return 0, 0
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    
    return epoch, global_step