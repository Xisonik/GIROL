import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
import gymnasium as gym
from skrl.utils.spaces.torch import (
    compute_space_size,
    flatten_tensorized_space,
    sample_space,
    unflatten_tensorized_space,
)

class GRUActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, **kwargs):
        super().__init__(observation_space, action_space, device)
        GaussianMixin.__init__(self, **kwargs)
        
        self.img_dim = observation_space["img"].shape[0]
        self.num_observations = self.img_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.LayerNorm(256),
            nn.ELU()
        )
        
        # GRU вместо LSTM - проще и часто работает лучше
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=512,
            num_layers=2,  # Можно использовать больше слоев
            batch_first=True,
            dropout=0.1
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, self.num_actions),
            nn.Tanh()
        )
        
        self.hidden = None
        
    def reset_hidden(self, batch_size=1):
        self.hidden = torch.zeros(2, batch_size, 512).to(self.device)  # 2 layers
    
    def compute(self, inputs, role):
        inputs_unflatten = unflatten_tensorized_space(self.observation_space, inputs["states"])
        x = inputs_unflatten["img"]
        
        encoded = self.encoder(x).unsqueeze(1)
        
        if self.hidden is None:
            self.reset_hidden(encoded.size(0))
        
        gru_out, self.hidden = self.gru(encoded, self.hidden)
        actions = self.decoder(gru_out[:, -1, :])
        
        return actions, self.log_std_parameter, {}