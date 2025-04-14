import torch
import torch.nn.functional as F
from typing import List, Tuple

def embed_roles_in_input(input_text: str, roles: str) -> str:
    """
    Append roles to the input string as a special [ROLES] section.
    """
    if roles and roles.lower() != "none":
        return f"{input_text} [ROLES] {roles}"
    return input_text

def format_prompt_with_roles(input_text: str, roles: str) -> str:
    """
    Format input as a structured prompt.
    Example: "Roles: agent(x,y); theme(x,z) | Input: The man gave the apple"
    """
    if roles and roles.lower() != "none":
        return f"Roles: {roles} | Input: {input_text}"
    return f"Input: {input_text}"

def compute_role_weighted_loss(
    loss: torch.Tensor,
    roles: List[str],
    base_weight: float = 1.0,
    role_weight: float = 2.0
) -> torch.Tensor:
    """
    Adjust batch loss by scaling samples with non-empty roles.
    Args:
        loss: Tensor of shape (batch,)
        roles: List of role strings (e.g. "agent(x,y); theme(x,z)")
        base_weight: Weight for empty-role examples
        role_weight: Weight for examples with roles
    """
    weights = []
    for role in roles:
        if role and role.strip().lower() != "none":
            weights.append(role_weight)
        else:
            weights.append(base_weight)

    weights_tensor = torch.tensor(weights, device=loss.device)
    scaled_loss = loss * weights_tensor
    return scaled_loss.mean()
