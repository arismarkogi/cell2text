"""
Configuration file for scGPT fine-tuning
"""
from pathlib import Path
import json

class Config:
    def __init__(self, config_dict=None):
        # Default hyperparameters
        self.seed = 0
        self.dataset_name = "custom"
        self.do_train = True
        self.load_model = "/home/arism/scgpt_model/scGPT_human"  # Path to pre-trained model
        
        # Training parameters
        self.mask_ratio = 0.0
        self.epochs = 3
        self.lr = 1e-4
        self.batch_size = 8
        self.eval_batch_size = 8
        
        # Model parameters
        self.n_bins = 51
        self.layer_size = 128
        self.nlayers = 4
        self.nhead = 4
        self.dropout = 0.2
        self.schedule_ratio = 0.9
        
        # Task-specific parameters
        self.freeze = True  # Freeze encoder weights
        self.classification_task = "cell_type"  # "celltype", "disease", or "tissue"
        
        # Other parameters
        self.MVC = False
        self.ecs_thres = 0.0
        self.dab_weight = 0.0
        self.save_eval_interval = 5
        self.fast_transformer = True
        self.pre_norm = False
        self.amp = True
        self.include_zero_gene = False
        self.DSBN = False
        
        # Data parameters
        self.max_seq_len = 3001
        self.input_style = "binned"  # "normed_raw", "log1p", or "binned"
        self.output_style = "binned"
        self.input_emb_style = "continuous"
        self.cell_emb_style = "cls"
        
        # Update with provided config
        if config_dict:
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    @classmethod
    def from_json(cls, json_path):
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    def to_json(self, json_path):
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_')}
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def validate(self):
        """Validate configuration parameters"""
        assert self.input_style in ["normed_raw", "log1p", "binned"]
        assert self.output_style in ["normed_raw", "log1p", "binned"]
        assert self.input_emb_style in ["category", "continuous", "scaling"]
        #assert self.classification_task in ["cell_type", "disease", "tissue"]
        
        if self.input_style == "binned" and self.input_emb_style == "scaling":
            raise ValueError("input_emb_style `scaling` is not supported for binned input.")
        elif self.input_style in ["log1p", "normed_raw"] and self.input_emb_style == "category":
            raise ValueError("input_emb_style `category` is not supported for log1p or normed_raw input.")

def get_task_specific_config(task="cell_type"):
    """Get task-specific configuration"""
    base_config = {
        "seed": 0,
        "epochs": 3,
        "lr": 1e-4,
        "batch_size": 8,
        "freeze": True,
        "classification_task": task
    }
    
    # Task-specific adjustments
    if task == "cell_type":
        base_config.update({
            "epochs": 3
            ,
            "lr": 1e-4,
        })
    elif task == "disease":
        base_config.update({
            "epochs": 3,
            "lr": 1e-4,
        })
    elif task == "tissue":
        base_config.update({
            "epochs": 3,
            "lr": 1e-4,
        })
    
    return Config(base_config)