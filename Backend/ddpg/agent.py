"""
The full DDPG agent: Actor, Critic, their slow-updating Target copies,
and the training step that ties them together via the replay buffer.

    1-2  state observed by Actor and Critic
    3    action proposed by the Actor
    4    Q-value / gradient returned by the Critic (learning signal)
    5    (s, a, r, s') stored in the Replay Buffer
    6    mini-batch sampled from the buffer
    7-8  soft update nudges Target Actor / Target Critic toward the mains
"""
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F

from ddpg.networks import Actor, Critic, ACTION_DIM


class DDPGAgent:
    def __init__(
        self,
        device: str = "cpu",
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.9,      # discount over the multi-step refinement passes
        tau: float = 0.005,      # soft-update rate for target networks
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    # ---- acting -----------------------------------------------------
    def select_action(self, state: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        """state: (3, H, W) float32 in [0, 1]. Returns action in [-1, 1]^7."""
        self.actor.eval()
        with torch.no_grad():
            s = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
            action = self.actor(s).cpu().numpy()[0]
        self.actor.train()
        if noise_std > 0:
            action = action + np.random.normal(0, noise_std, size=ACTION_DIM)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    # ---- learning -----------------------------------------------------
    def train_step(self, batch, batch_size: int = 64):
        if len(batch) < batch_size:
            return None
        states, actions, rewards, next_states, dones = batch.sample(batch_size)

        s = torch.from_numpy(states).float().to(self.device)
        a = torch.from_numpy(actions).float().to(self.device)
        r = torch.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        s2 = torch.from_numpy(next_states).float().to(self.device)
        d = torch.from_numpy(dones).float().unsqueeze(1).to(self.device)

        # --- Critic update: minimize TD error against the target networks ---
        with torch.no_grad():
            next_actions = self.target_actor(s2)
            target_q = self.target_critic(s2, next_actions)
            y = r + self.gamma * (1 - d) * target_q
        q = self.critic(s, a)
        critic_loss = F.mse_loss(q, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # --- Actor update: deterministic policy gradient ---
        # Maximize Q(s, mu(s))  <=>  minimize -Q(s, mu(s)).
        # This is how a gradient flows back to the Actor even though the
        # perceptual reward itself (UIQM/UCIQE/PSNR/SSIM) is non-differentiable.
        actor_loss = -self.critic(s, self.actor(s)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}

    def soft_update(self, target: torch.nn.Module, source: torch.nn.Module):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tp.data * (1.0 - self.tau) + sp.data * self.tau)

    # ---- persistence -----------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "target_critic": self.target_critic.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.target_actor.load_state_dict(ckpt["target_actor"])
        self.target_critic.load_state_dict(ckpt["target_critic"])
