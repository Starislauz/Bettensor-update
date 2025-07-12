import torch
import torch.nn as nn
import os

# Define the same KellyFractionNet architecture as in model_utils.py
class KellyFractionNet(nn.Module):
    def __init__(self, input_size):
        super(KellyFractionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc4(x)) * 0.5

# Create model
model = KellyFractionNet(input_size=50)

# Save it as a state dict
model_dir = "mlb_wager_model.pt"
os.makedirs(model_dir, exist_ok=True)

# Save the model state dictionary
torch.save(model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))

# Create a simple config file
config = {
    "input_size": 50,
    "model_type": "KellyFractionNet",
    "_name_or_path": "mlb_wager_model"
}

import json
with open(os.path.join(model_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"MLB model saved to {model_dir}/")
print("Files created:")
for file in os.listdir(model_dir):
    print(f"  - {file}")
