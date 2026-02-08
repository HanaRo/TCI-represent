import torch
import torch.nn as nn

class ScoreNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=[64]):
        super().__init__()
        assert isinstance(hidden_dim, list) and len(hidden_dim) > 0, "hidden_dim should be a non-empty list"
        # construct the network based on the hidden_dim
        modules = []
        dim_in = input_dim
        for dim_out in hidden_dim:
            modules.append(nn.Linear(dim_in, dim_out))
            modules.append(nn.ReLU())
            dim_in = dim_out
        modules.append(nn.Linear(dim_in, 1))
        self.f = nn.Sequential(*modules)

    def forward(self, x):
        return self.f(x).squeeze(-1)
    
class ScoreNetSig(nn.Module):
    def __init__(self, input_dim, hidden_dim=[64]):
        super().__init__()
        assert isinstance(hidden_dim, list) and len(hidden_dim) > 0, "hidden_dim should be a non-empty list"
        # construct the network based on the hidden_dim
        modules = []
        dim_in = input_dim
        for dim_out in hidden_dim:
            modules.append(nn.Linear(dim_in, dim_out))
            modules.append(nn.ReLU())
            dim_in = dim_out
        modules.append(nn.Linear(dim_in, 1))
        modules.append(nn.Sigmoid())
        self.f = nn.Sequential(*modules)

    def forward(self, x):
        return self.f(x).squeeze(-1)
    
if __name__ == "__main__":
    model = ScoreNet(input_dim=32)
    x = torch.randn(10, 32)  # Example input
    output = model(x)
    print(output.shape)  # Should print torch.Size([10])