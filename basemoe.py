# simple expert choice MoE

import torch
import torch.nn as nn
import torch.nn.functional as F

# config
# moe_num_experts = 4
# config.model_dim
# config.capacity_factor
# config.moe_softmax_order
# config.batch_size
# config.seq_len


class SimpleExpertChoiceMoE(nn.Module):
    """
    Simple MoE implementation with Expert Choice routing
    """

    def __init__(self, config, mlp):
        super().__init__()

        self.num_experts = config.moe_num_experts
        self.experts = nn.ModuleList(
            [mlp(config=config) for _ in range(self.num_experts)]
        )
        self.router = nn.Linear(config.model_dim, self.num_experts, bias=False)
        self.capacity_factor = config.capacity_factor
        self.softmax_order = config.moe_softmax_order
        self.top_k = int(
            self.capacity_factor * config.batch_size * config.seq_len / self.num_experts
        )

    def forward(self, inputs: torch.Tensor):
        # [batch_size * sequence_length, model_dim]
        inputs_squashed = inputs.view(-1, inputs.shape[-1])

        num_tokens = inputs_squashed.shape[0]

        top_k = min(
            self.top_k, int(self.capacity_factor * num_tokens / self.num_experts)
        )

        # [batch_size * sequence_length, num_experts]
        router_output = self.router(inputs_squashed)
        # [batch_size * sequence_length
