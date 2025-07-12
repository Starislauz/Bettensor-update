import torch
import torch.nn as nn
import os
import sys

# Add parent directories to path to import KellyFractionNet
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from model_utils import KellyFractionNet

def create_mlb_wager_model():
    """Create a mock MLB wager model for testing"""
    # Create model with 50 input features (matching our mock preprocessor)
    model = KellyFractionNet(input_size=50)
    
    # Initialize with some reasonable weights
    with torch.no_grad():
        for param in model.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
    
    # Save the model
    model_dir = "mlb_wager_model.pt"
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    print(f"MLB wager model saved to {model_dir}")
    
    return model

if __name__ == "__main__":
    create_mlb_wager_model()
