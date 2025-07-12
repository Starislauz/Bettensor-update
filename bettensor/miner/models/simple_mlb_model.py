import torch
import os
import sys

# Import the KellyFractionNet from model_utils
sys.path.append(os.path.dirname(__file__))
from model_utils import KellyFractionNet

def create_mlb_model():
    # Create a simple model with 50 input features
    model = KellyFractionNet(input_size=50)
    
    # Create directory for the model
    model_dir = "mlb_wager_model.pt"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save model components manually
    config = {
        "input_size": 50,
        "model_type": "KellyFractionNet"
    }
    
    # Save config
    import json
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f)
    
    # Save model state dict
    torch.save(model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))
    
    print(f"MLB model saved to {model_dir}")

if __name__ == "__main__":
    create_mlb_model()
