import torch
import torch.nn as nn

GESTURE_CLASSES = ["neutral", "point", "pinch", "open_palm", "fist", "two_fingers"]

class GestureMLP(nn.Module):
    def __init__(self, input_dim=63, num_classes = 6):
        super().__init__()
        self.net= nn.Sequential(
            nn.Linear(input_dim, 128),nn.ReLU(),
            nn.Linear(128, 64),nn.ReLU(),
            nn.Linear(64, 32),nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def load_model(path = "models/gesture_mlp.pt", device = "cpu"):
    model = GestureMLP()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

