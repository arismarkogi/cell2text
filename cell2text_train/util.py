import argparse
from  tqdm import tqdm
import sys
import numpy as np
import json
import random
import os
from torch.utils.data import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cell2text_dataset.dataset import Cell2TextDataset


def create_progress_bar(description, total, is_main_process):
    """Create progress bar only on main process"""
    if is_main_process:
        return tqdm(
            desc=description, 
            total=total,
            position=0,
            leave=True,
            file=sys.stdout
        )
    else:
        return None


def create_argument_parser():
    """Create and return the argument parser for DeepSpeed training"""
    parser = argparse.ArgumentParser(description="DeepSpeed training for Cell2Text model")

     # Add experiment naming parameter
    parser.add_argument("--experiment_name", type=str, required=True,
                        help="Name for the experiment (will be used in output directory and file names)")
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=["sanity", "full"], default="sanity",
                        help="Training mode: 'sanity' for sanity check or 'full' for full training")
    
    # Data parameters
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to the parquet file containing training data")
    parser.add_argument("--val_data_path", type=str, default=None,
                        help="Path to the parquet file containing validation data (for full training)")
    parser.add_argument("--output_dir", type=str, default="./deepspeed_training",
                        help="Directory to save training results")
    
    # Sanity check specific parameters
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Number of samples to overfit on")
    parser.add_argument("--target_loss", type=float, default=0.01,
                        help="Target loss to reach (default: 0.01)")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Training epochs")
    parser.add_argument("--save_overfitted_model", type=bool, default=True,
                        help="Save the overfitted model")
    parser.add_argument("--save_results", type=bool, default=True,
                        help="Save evaluation results to JSON")
    
    # Model parameters
    parser.add_argument("--encoder_hidden_size", type=int, default=1152,
                        help="Hidden size of the cell encoder")
    parser.add_argument("--mlp_hidden_size", type=int, default=1024,
                        help="Hidden size of the 2-layer MLP cell-to-embedding projector")
    parser.add_argument("--mlp_dropout", type=float, default=0.1, 
                        help="Dropout probability at MLP projector")
    parser.add_argument("--decoder_hidden_size", type=int, default=2048,
                        help="Hidden size of the decoder")
    parser.add_argument("--top_k", type=int, default=384,
                        help="select the top_k most expressed genes after the Geneformer encoder")
    parser.add_argument("--geneformer_path", type=str, required=True,
                        help="Path to pretrained Geneformer model")
    parser.add_argument("--decoder_path", type=str, required=True,
                        help="Path to pretrained Llama decoder model")
    parser.add_argument("--token_dictionary_path", type=str, required=True,
                        help="Path to geneformer Dictionary file")
    parser.add_argument("--max_ncells", type=int, default=1000,
                        help="Maximum number of cells")
    parser.add_argument("--max_length", type=int, default=500,
                        help="Maximum length of generated text")
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Number of beams for beam search")
    
    # LoRA parameters
    parser.add_argument("--use_lora_decoder", type=bool, default=True,
                        help="Use LoRA for the text decoder (LLaMA)")
    
    # LoRA parameters for decoder
    parser.add_argument("--lora_r_decoder", type=int, default=16,
                        help="LoRA rank for decoder")
    parser.add_argument("--lora_alpha_decoder", type=int, default=32,
                        help="LoRA alpha for decoder")
    parser.add_argument("--lora_dropout_decoder", type=float, default=0.1,
                        help="LoRA dropout for decoder")
    parser.add_argument("--lora_target_modules_decoder", type=str, nargs="+", 
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        help="Target modules for LoRA in decoder (LLaMA modules)")
    parser.add_argument("--lora_bias_decoder", type=str, default="none",
                        choices=["none", "all", "lora_only"],
                        help="LoRA bias type for decoder")
    parser.add_argument("--lora_modules_to_save_decoder", type=str, nargs="*", default=None,
                        help="Additional modules to save for decoder LoRA")
    
    # Freezing parameters
    parser.add_argument("--freeze_decoder", type=bool, default=True,
                        help="Freeze decoder parameters (only applies if not using LoRA for decoder)")
    
    # DeepSpeed specific parameters
    parser.add_argument("--batch_size_per_device", type=int, default=8,
                        help="Batch size per device/GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--zero_stage", type=int, default=2, choices=[0, 1, 2, 3],
                        help="DeepSpeed ZeRO optimization stage")
    parser.add_argument("--fp16", type=bool, default=False,
                        help="Enable FP16 mixed precision training")
    parser.add_argument("--bf16", type=bool, default=False,
                        help="Enable BF16 mixed precision training")
    
    # Training parameters (optimized for sanity check)
    parser.add_argument("--decoder_lr", type=float, default=1e-4,
                        help="Learning rate for the decoder (higher for faster overfitting)")
    parser.add_argument("--projector_lr", type=float, default=1e-3,
                        help="Learning rate for the projection layer (higher for faster overfitting)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay for AdamW optimizer (0 for easier overfitting)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--warmup_steps", type=int, default=10,
                        help="Number of warmup steps for learning rate schedule")

    # Misc parameters
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (automatically set by DeepSpeed)")
    
    # Projector type selection
    parser.add_argument("--projector", type=str, choices=["mlp", "perceiver"], default="mlp",
                        help="Type of projector to use: 'mlp' or 'perceiver'")
    
    # Perceiver-specific parameters
    parser.add_argument("--num_latents", type=int, default=64,
                        help="Number of latent vectors for Perceiver")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads in Perceiver")
    parser.add_argument("--ff_mult", type=float, default=4,
                        help="Feedforward multiplier in Perceiver")
    parser.add_argument("--perceiver_dropout", type=float, default=0.1,
                        help="Dropout probability in Perceiver")
    
    # Full training specific parameters
    parser.add_argument("--eval_steps", type=int, default=2000,
                        help="Number of steps between evaluations")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Number of steps between saving checkpoints")
    parser.add_argument("--early_stopping", type=int, default=0,
                        help="Number of steps without improvement before early stopping (0 to disable)")
    parser.add_argument("--save_model", type=bool, default=True,
                        help="Save model checkpoints")
    
    return parser


def compute_enhanced_training_summary(enhanced_training_history, enhanced_validation_history):
        """Compute enhanced summary statistics from training and validation history"""
        summary = {
            'training_stats': {},
            'validation_stats': {},
            'best_metrics': {},
            'training_progression': {},
            'validation_progression': {}
        }
        
        # Enhanced training stats
        if enhanced_training_history.get('loss_progression'):
            loss_stats = enhanced_training_history['loss_progression']['loss_statistics']
            summary['training_stats'] = loss_stats.copy()
            
            # Add progression analysis
            if enhanced_training_history['loss_progression']['all_losses']:
                losses = enhanced_training_history['loss_progression']['all_losses']
                summary['training_progression'] = {
                    'loss_trend': 'decreasing' if losses[-1] < losses[0] else 'increasing',
                    'loss_reduction_percentage': ((losses[0] - losses[-1]) / losses[0] * 100) if losses[0] != 0 else 0,
                    'convergence_point': find_convergence_point(losses)
                }
        
        # Enhanced validation stats
        if enhanced_validation_history.get('metric_progressions'):
            metric_progs = enhanced_validation_history['metric_progressions']
            
            summary['validation_stats'] = {}
            summary['validation_progression'] = {}
            summary['best_metrics'] = {}
            
            for metric, data in metric_progs.items():
                stats = data['statistics']
                summary['validation_stats'][f'{metric}_stats'] = stats
                summary['best_metrics'][f'best_{metric}'] = stats['best_value']
                
                # Progression analysis
                if len(data['all_values']) > 1:
                    values = data['all_values']
                    summary['validation_progression'][f'{metric}_trend'] = {
                        'direction': 'improving' if stats['improvement'] > 0 and 'loss' not in metric else 
                                   'improving' if stats['improvement'] < 0 and 'loss' in metric else 'degrading',
                        'improvement_percentage': abs(stats['improvement'] / values[0] * 100) if values[0] != 0 else 0,
                        'stability': 'stable' if stats['std_value'] < (stats['avg_value'] * 0.1) else 'unstable'
                    }
        
        return summary
    
def find_convergence_point( losses, window_size=100, threshold=0.01):
    """Find the point where loss starts to converge"""
    if len(losses) < window_size * 2:
            return None
        
    for i in range(window_size, len(losses) - window_size):
        window1 = losses[i-window_size:i]
        window2 = losses[i:i+window_size]
            
        avg1 = np.mean(window1)
        avg2 = np.mean(window2)
            
        if abs(avg1 - avg2) / avg1 < threshold:
            return i
        
    return None

def convert_json_compat(obj):
    if isinstance(obj, dict):
        return {k: convert_json_compat(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_json_compat(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj
    

class SanityDataset(Dataset):
    """Wrapper to create a small subset of data for sanity check"""
    
    def __init__(self, full_dataset, num_samples=8, seed=42):
        self.full_dataset = full_dataset
        self.num_samples = min(num_samples, len(full_dataset))
        
        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Select random indices
        self.indices = random.sample(range(len(full_dataset)), self.num_samples)
        print(f"Selected {self.num_samples} samples for sanity check: {self.indices}")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.full_dataset[self.indices[idx]]
    


def save_training_history(model, training_history, validation_history):
        """Save training and validation history to files with enhanced metrics tracking"""
        if  model.is_main_process:
            # Enhanced training history with loss progression
            enhanced_training_history = {
                'training_steps': training_history,
                'loss_progression': {
                    'all_losses': model.losses,
                    'loss_statistics': {
                        'initial_loss': model.losses[0] if model.losses else None,
                        'final_loss': model.losses[-1] if model.losses else None,
                        'min_loss': min(model.losses) if model.losses else None,
                        'max_loss': max(model.losses) if model.losses else None,
                        'avg_loss': np.mean(model.losses) if model.losses else None,
                        'loss_std': np.std(model.losses) if model.losses else None,
                        'total_steps': len(model.losses)
                    }
                }
            }
            
            # Enhanced validation history with metric progressions
            enhanced_validation_history = {
                'validation_steps': validation_history,
                'metric_progressions': {}
            }
            
            if validation_history:
                # Extract metric progressions
                metrics_to_track = [
                    'validation_loss', 'bleu_score', 'cell_type_accuracy', 
                    'cell_type_f1', 'cell_type_precision', 'cell_type_recall'
                ]
                
                for metric in metrics_to_track:
                    values = [entry.get(metric) for entry in validation_history if entry.get(metric) is not None]
                    if values:
                        enhanced_validation_history['metric_progressions'][metric] = {
                            'all_values': values,
                            'statistics': {
                                'initial_value': values[0],
                                'final_value': values[-1],
                                'best_value': min(values) if 'loss' in metric else max(values),
                                'worst_value': max(values) if 'loss' in metric else min(values),
                                'avg_value': np.mean(values),
                                'std_value': np.std(values),
                                'total_evaluations': len(values),
                                'improvement': values[-1] - values[0] if len(values) > 1 else 0
                            }
                        }
            
            # Save enhanced histories
            training_history_path = os.path.join(model.args.output_dir, "enhanced_training_history.json")
            with open(training_history_path, 'w') as f:
                json.dump(convert_json_compat(enhanced_training_history), f, indent=2)
            
            validation_history_path = os.path.join(model.args.output_dir, "enhanced_validation_history.json")
            with open(validation_history_path, 'w') as f:
                json.dump(convert_json_compat(enhanced_validation_history), f, indent=2)
            
            # Save summary statistics (enhanced version)
            summary_stats = compute_enhanced_training_summary(enhanced_training_history, enhanced_validation_history)
            summary_path = os.path.join(model.args.output_dir, "enhanced_training_summary_stats.json")
            with open(summary_path, 'w') as f:
                json.dump(convert_json_compat(summary_stats), f, indent=2)
            
            print(f"Enhanced training history saved to: {training_history_path}")
            print(f"Enhanced validation history saved to: {validation_history_path}")
            print(f"Enhanced training summary stats saved to: {summary_path}")