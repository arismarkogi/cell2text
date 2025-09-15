import torch
import torch.nn as nn
from classifier import GeneformerPathwayClassifier as PathwayClassifier

# 1. Define the model class (assuming it's imported correctly).
# from classifier import GeneformerPathwayClassifier as PathwayClassifier

# 2. Instantiate the model.
# Create an instance of the model class you defined above.
model = PathwayClassifier(geneformer_model_path="/home/arism/cell2text/Geneformer", num_pathways=34, freeze_geneformer=True)

# 3. Load the state_dict from the .pt file.
# We use map_location='cpu' to ensure it loads correctly even if the file was saved on a GPU.
try:
    state_dict = torch.load(
        '/home/arism/pathway_results/2025-09-14_ddp_best_model.pt',
        map_location='cpu',
        weights_only=False  # This can be set to True if you only want weights, but False is more flexible.
    )['model_state_dict']

    # 4. Load the weights into the model instance.
    # This is the crucial missing step.
    model.load_state_dict(state_dict)
    
    # 5. Inspect the weights after loading.
    print("Model parameters (weights and biases) after loading state_dict:")
    print("-" * 50)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}")
            print(f"Shape: {param.data.shape}")
            # You might not want to print all values, as it can be very large.
            # A good practice is to print the first few values or summary statistics.
            print(f"First 5 values:\n{param.data.flatten()[:5]}\n")

except FileNotFoundError:
    print(f"Error: The file 'best_model.pt' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")