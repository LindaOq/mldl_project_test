from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        
        # Define layers of the neural network
        self.layer_stack = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.Conv2d(256, 64, kernel_size=3, padding=1),
        nn.Flatten(),
        nn.Linear(64*64*64, 200),
        nn.ReLU()) # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        # Define forward pass
        x = self.layer_stack(x)

        return x
