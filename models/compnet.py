import torch
import torch.nn as nn
import torch.functional as F

class LearnableActivation(nn.Module):
    def __init__(self, alpha_init=0.01, beta_init=10.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))
    def forward(self, x):
        return torch.log(1 + self.alpha * torch.exp(self.beta * x))

class CompNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=[64], comp_dims=[[128, 128], [128, 128]]):
        super().__init__()
        assert isinstance(hidden_dim, list) and len(hidden_dim) > 0, "hidden_dim should be a non-empty list"
        # construct the network based on the hidden_dim (task)
        t_modules = []
        dim_in = input_dim
        for dim_out in hidden_dim:
            t_modules.append(nn.Linear(dim_in, dim_out))
            t_modules.append(nn.ReLU())
            dim_in = dim_out
        t_modules.append(nn.Linear(dim_in, 1))
        self.t_f = nn.Sequential(*t_modules)
        # construct the network based on the hidden_dim (capability)
        c_modules = []
        dim_in = input_dim
        for dim_out in hidden_dim:
            c_modules.append(nn.Linear(dim_in, dim_out))
            c_modules.append(nn.ReLU())
            dim_in = dim_out
        c_modules.append(nn.Linear(dim_in, 1))
        self.c_f = nn.Sequential(*c_modules)
        # construct the network based on the comp_dim
        assert isinstance(comp_dims, list) and all(isinstance(d, list) for d in comp_dims), "comp_dim should be a list of lists"
        self.comp_f = nn.ModuleList()
        for _, comp_dim in enumerate(comp_dims):
            comp_modules = []
            dim_in = 2 * input_dim + 2 # +2 for task and capability scores
            for dim_out in comp_dim:
                comp_modules.append(nn.Linear(dim_in, dim_out))
                comp_modules.append(nn.ReLU())
                dim_in = dim_out
            comp_modules.extend((
                nn.Linear(dim_in, 1),
                nn.Sigmoid()  # Use Sigmoid for compatibility score
            ))
            comp_sub_f = nn.Sequential(*comp_modules)
            self.comp_f.append(comp_sub_f)

    def forward(self, x):
        t , c = x['task'], x['capability']
        t_score = self.t_f(t)  # Task score
        c_score = self.c_f(c)  # Capability score
        # Concatenate task and capability scores with the input
        comp_input = torch.cat([t, c, t_score, c_score], dim=-1)
        comp_score = []
        for comp_sub_f in self.comp_f:
            comp_score.append(comp_sub_f(comp_input))
        comp_score = torch.stack(comp_score, dim=1)  # Stack compatibility scores from
        return {
            'task_score': t_score.squeeze(-1),
            'capability_score': c_score.squeeze(-1),    
            'compatibility_score': comp_score.squeeze(-1)  # Shape: [batch_size, num_comp_scores]
        }
    
class CompNetPlus(nn.Module):
    def __init__(self, input_dim, hidden_dim=[64, 64], comp_num=[[0.01, 10]]):
        super().__init__()
        assert isinstance(hidden_dim, list) and len(hidden_dim) > 0, "hidden_dim should be a non-empty list"
        # construct the network based on the hidden_dim (task)
        t_modules = []
        dim_in = input_dim
        for dim_out in hidden_dim:
            t_modules.append(nn.Linear(dim_in, dim_out))
            t_modules.append(nn.ReLU())
            dim_in = dim_out
        t_modules.append(nn.Linear(dim_in, 1))
        self.t_f = nn.Sequential(*t_modules)
        # construct the network based on the hidden_dim (capability)
        c_modules = []
        dim_in = input_dim
        for dim_out in hidden_dim:
            c_modules.append(nn.Linear(dim_in, dim_out))
            c_modules.append(nn.ReLU())
            dim_in = dim_out
        c_modules.append(nn.Linear(dim_in, 1))
        self.c_f = nn.Sequential(*c_modules)
        self.comp_f = nn.ModuleList()
        for alpha, beta in comp_num:
            self.comp_f.append(LearnableActivation(alpha, beta))

    def forward(self, x):
        t , c = x['task'], x['capability']
        t_score = self.t_f(t)  # Task score
        c_score = self.c_f(c)  # Capability score
        # Concatenate task and capability scores with the input
        comp_input = t_score - c_score
        comp_score = []
        for comp_sub_f in self.comp_f:
            score = comp_sub_f(comp_input)
            score = torch.clamp(score, 0, 1)  # Ensure compatibility scores are in [0, 1]
            comp_score.append(score)
        comp_score = torch.stack(comp_score, dim=1)  # Stack compatibility scores from
        return {
            'task_score': t_score.squeeze(-1),
            'capability_score': c_score.squeeze(-1),    
            'compatibility_score': comp_score.squeeze(-1)  # Shape: [batch_size, num_comp_scores]
        }
    
class CompNetPlusPlus(nn.Module):
    def __init__(self, input_dim, hidden_dim=[64, 64], comp_dim=[64, 64]):
        super().__init__()
        assert isinstance(hidden_dim, list) and len(hidden_dim) > 0, "hidden_dim should be a non-empty list"
        # construct the network based on the hidden_dim (task)
        t_modules = []
        dim_in = input_dim
        for dim_out in hidden_dim:
            t_modules.append(nn.Linear(dim_in, dim_out))
            t_modules.append(nn.ReLU())
            dim_in = dim_out
        t_modules.append(nn.Linear(dim_in, 1))
        self.t_f = nn.Sequential(*t_modules)
        # construct the network based on the hidden_dim (capability)
        c_modules = []
        dim_in = input_dim
        for dim_out in hidden_dim:
            c_modules.append(nn.Linear(dim_in, dim_out))
            c_modules.append(nn.ReLU())
            dim_in = dim_out
        c_modules.append(nn.Linear(dim_in, 1))
        self.c_f = nn.Sequential(*c_modules)
        self.comp_f = nn.ModuleList()
        comp_modules = []
        dim_in = 2
        for dim_out in comp_dim:
            comp_modules.append(nn.Linear(dim_in, dim_out))
            comp_modules.append(nn.ReLU())
            dim_in = dim_out
        comp_modules.append(nn.Linear(dim_in, 1))
        # comp_modules.append(nn.Sigmoid())  # Use Sigmoid for compatibility score
        self.comp_f = nn.Sequential(*comp_modules)

    def forward(self, x):
        t , c = x['task'], x['capability']
        t_score = self.t_f(t)  # Task score
        c_score = self.c_f(c)  # Capability score
        # Concatenate task and capability scores with the input
        comp_input = torch.cat([t_score, c_score], dim=-1)  # Shape: [batch_size, 2]
        comp_score = self.comp_f(comp_input)
        # comp_score = torch.clamp(comp_score, 0, 1)  # Ensure compatibility scores are in [0, 1]
        return {
            'task_score': t_score.squeeze(-1),
            'capability_score': c_score.squeeze(-1),    
            'compatibility_score': comp_score.squeeze(-1)  # Shape: [batch_size, num_comp_scores]
        }
    
class CompNext(nn.Module):
    def __init__(self, input_dim, hidden_dim=[64, 64], comp_dim=[64, 64]):
        super().__init__()
        assert isinstance(hidden_dim, list) and len(hidden_dim) > 0, "hidden_dim should be a non-empty list"
        # construct the network based on the hidden_dim (task)
        t_modules = []
        dim_in = input_dim
        for dim_out in hidden_dim:
            t_modules.append(nn.Linear(dim_in, dim_out))
            t_modules.append(nn.ReLU())
            dim_in = dim_out
        t_modules.append(nn.Linear(dim_in, 1))
        self.t_f = nn.Sequential(*t_modules)
        # construct the network based on the hidden_dim (capability)
        c_modules = []
        dim_in = input_dim
        for dim_out in hidden_dim:
            c_modules.append(nn.Linear(dim_in, dim_out))
            c_modules.append(nn.ReLU())
            dim_in = dim_out
        c_modules.append(nn.Linear(dim_in, 1))
        self.c_f = nn.Sequential(*c_modules)
        self.comp_f = nn.ModuleList()
        comp_modules = []
        dim_in = input_dim * 2 + 2
        for dim_out in comp_dim:
            comp_modules.append(nn.Linear(dim_in, dim_out))
            comp_modules.append(nn.ReLU())
            dim_in = dim_out
        comp_modules.append(nn.Linear(dim_in, 1))
        # comp_modules.append(nn.Sigmoid())  # Use Sigmoid for compatibility score
        self.comp_f = nn.Sequential(*comp_modules)

    def forward(self, x):
        t , c = x['task'], x['capability']
        t_score = self.t_f(t)  # Task score
        c_score = self.c_f(c)  # Capability score
        # Concatenate task and capability scores with the input
        comp_input = torch.cat([t, c, t_score, c_score], dim=-1)  # Shape: [batch_size, input_dim * 2 + 2]
        comp_score = self.comp_f(comp_input)
        # comp_score = torch.clamp(comp_score, 0, 1)  # Ensure compatibility scores are in [0, 1]
        return {
            'task_score': t_score.squeeze(-1),
            'capability_score': c_score.squeeze(-1),    
            'compatibility_score': comp_score.squeeze(-1)  # Shape: [batch_size, num_comp_scores]
        }

if __name__ == "__main__":
    model = CompNetPlus(input_dim=32, comp_num=2)
    x = {
        'task': torch.randn(10, 32),  # Example task input
        'capability': torch.randn(10, 32)  # Example capability input
    }
    output = model(x)
    print(output['task_score'].shape)  # Should print torch.Size([10])
    print(output['capability_score'].shape)  # Should print torch.Size([10])
    print(output['compatibility_score'].shape)  # Should print torch.Size([10])
