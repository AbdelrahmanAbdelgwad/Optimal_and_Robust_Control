import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
from typing import Tuple
import random
from collections import deque
import os
from datetime import datetime
import json

# Set up CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SafetyQNetwork(nn.Module):
    """
    Neural network for safety value prediction Q(s,a) ∈ [0,1].
    The output represents the probability of maintaining safety when taking action a in state s
    and acting optimally thereafter.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Main network architecture - designed for continuous state-action spaces
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ).to(device)

        self.action_net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ).to(device)

        self.combined_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Bound output to [0,1] for safety probability
        ).to(device)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute safety value Q(s,a) for given state-action pairs.
        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
        Returns:
            Safety values [batch_size, 1]
        """
        state_features = self.state_net(state)
        action_features = self.action_net(action)
        combined = torch.cat([state_features, action_features], dim=1)
        return self.combined_net(combined)

    def compute_safety_integral(
        self, state: torch.Tensor, action_space: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the integral of Q(s,a) over all actions by iterating over discrete actions.
        This estimates E_a[Q(s,a)] for the given state.

        Args:
            state: [batch_size, state_dim]
            action_space: Tensor of all possible actions [num_actions, action_dim]

        Returns:
            Expected safety value [batch_size, 1]
        """
        batch_size = state.shape[0]
        num_actions = action_space.shape[0]

        # Expand states to match action samples
        expanded_states = state.unsqueeze(1).expand(-1, num_actions, -1)

        # Reshape for batch processing
        flat_states = expanded_states.reshape(-1, expanded_states.shape[-1])
        flat_actions = action_space.expand(batch_size, -1, -1).reshape(
            -1, action_space.shape[-1]
        )

        # Compute Q-values for all state-action pairs
        q_values = self.forward(flat_states, flat_actions)

        # Reshape back and compute mean over actions
        q_values = q_values.reshape(batch_size, num_actions)
        return q_values.mean(dim=1, keepdim=True)


class SafetyReplayBuffer:
    def __init__(self, capacity: int = int(1e6)):
        self.buffer = deque(maxlen=capacity)
        self.collision_indices = (
            []
        )  # Track collision transitions for importance sampling

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        collision: bool,
    ):
        self.buffer.append((state, action, next_state, collision))

        if collision:
            self.collision_indices.append(len(self.buffer) - 1)

        # Remove outdated collision indices
        while self.collision_indices and self.collision_indices[0] >= len(self.buffer):
            self.collision_indices.pop(0)

    def sample(self, batch_size: int, collision_prob: float = 0.3) -> tuple:
        indices = []

        # Sample collision transitions
        n_collision = int(batch_size * collision_prob)
        if self.collision_indices and n_collision > 0:
            collision_samples = min(n_collision, len(self.collision_indices))
            indices.extend(random.sample(self.collision_indices, collision_samples))

        # Sample regular transitions
        n_regular = batch_size - len(indices)
        if n_regular > 0:
            regular_indices = [
                i for i in range(len(self.buffer)) if i not in self.collision_indices
            ]
            indices.extend(random.sample(regular_indices, n_regular))

        # Gather batch
        batch = [self.buffer[idx] for idx in indices]
        states, actions, next_states, collisions = zip(*batch)

        # Convert to tensors and move to device
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        collisions = torch.BoolTensor(collisions).to(device)

        return states, actions, next_states, collisions

    def __len__(self) -> int:
        return len(self.buffer)


class SafetyLearner:
    def __init__(
        self,
        env: gym.Env,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        learning_rate: float = 3e-4,
        buffer_size: int = int(1e6),
        batch_size: int = 64,
        action_limit: float = 1.0,
        checkpoint_dir: str = "checkpoints",
    ):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_limit = action_limit
        self.checkpoint_dir = checkpoint_dir

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize checkpoint tracking
        self.best_safety_value = float("-inf")
        self.checkpoint_counter = 0

        # Training metrics history
        self.metrics_history = {
            "episode_lengths": [],
            "collision_rates": [],
            "losses": [],
            "safety_values": [],
            "best_safety_value": float("-inf"),
        }

        # Initialize networks
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.q_network = SafetyQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_network = SafetyQNetwork(state_dim, action_dim, hidden_dim).to(
            device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Initialize replay buffer
        self.replay_buffer = SafetyReplayBuffer(buffer_size)

        # Training metrics
        self.steps = 0
        self.collision_count = 0

    def save_checkpoint(self, checkpoint_type: str, metrics: dict, episode: int):
        """
        Save model checkpoint and training metrics.

        Args:
            checkpoint_type: Type of checkpoint ('periodic' or 'best')
            metrics: Current training metrics
            episode: Current episode number
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create checkpoint filename
        if checkpoint_type == "periodic":
            filename = f"checkpoint_periodic_ep{episode}_{timestamp}"
        else:
            filename = f"checkpoint_best_ep{episode}_{timestamp}"

        # Save model state
        checkpoint = {
            "episode": episode,
            "model_state_dict": self.q_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "steps": self.steps,
            "best_safety_value": self.best_safety_value,
        }

        # Save checkpoint
        model_path = os.path.join(self.checkpoint_dir, f"{filename}.pth")
        torch.save(checkpoint, model_path)

        # Save metrics separately for easier analysis
        metrics_path = os.path.join(self.checkpoint_dir, f"{filename}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Saved {checkpoint_type} checkpoint at episode {episode}")

    def update(self) -> float:
        """
        Implement the safety update law:
        if no collision:
            Q(s,a) = 1-gamma + gamma * E_a'[Q(s',a')]
        if collision:
            Q(s,a) = 0
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample transitions
        states, actions, next_states, collisions = self.replay_buffer.sample(
            self.batch_size
        )

        # Compute target values
        with torch.no_grad():
            # Compute expected future safety for non-collision states
            future_safety = self.target_network.compute_safety_integral(next_states)

            # Implement the safety update law
            target_values = torch.where(
                collisions,
                torch.zeros_like(future_safety, device=device),  # Collision case
                1 - self.gamma + self.gamma * future_safety,  # Safe case
            )

        # Compute current safety predictions
        current_values = self.q_network(states, actions)

        # Compute loss and optimize
        loss = F.mse_loss(current_values, target_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        if self.steps % 100 == 0:
            for target_param, param in zip(
                self.target_network.parameters(), self.q_network.parameters()
            ):
                target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

        return loss.item()

    def train(
        self,
        n_episodes: int,
        max_steps: int = 1000,
        checkpoint_freq: int = 100,  # Save checkpoint every N episodes
        eval_freq: int = 10,  # Evaluate and potentially save best model every N episodes
    ) -> dict:
        metrics = {
            "episode_lengths": [],
            "collision_rates": [],
            "losses": [],
            "safety_values": [],
        }

        for episode in range(n_episodes):
            state, _ = self.env.reset()
            episode_length = 0
            episode_loss = 0
            episode_safety_values = []

            for step in range(max_steps):
                # Select and execute action
                action = self.env.action_space.sample()

                # Record safety value
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
                    safety_value = self.q_network(state_tensor, action_tensor)
                    episode_safety_values.append(safety_value.cpu().item())

                # Execute action
                next_state, _, terminated, truncated, _ = self.env.step(action)
                collision = terminated and not truncated

                # Store transition
                self.replay_buffer.push(state, action, next_state, collision)

                # Update networks
                loss = self.update()
                episode_loss += loss

                state = next_state
                episode_length += 1
                self.steps += 1

                if terminated or truncated:
                    break

            # Update metrics
            metrics["episode_lengths"].append(episode_length)
            collision_rate = (
                len(self.replay_buffer.collision_indices) / len(self.replay_buffer)
                if len(self.replay_buffer) > 0
                else 0
            )
            metrics["collision_rates"].append(collision_rate)
            metrics["losses"].append(
                episode_loss / episode_length if episode_length > 0 else 0
            )

            avg_safety_value = np.mean(episode_safety_values)
            metrics["safety_values"].append(avg_safety_value)

            # Save periodic checkpoint
            if (episode + 1) % checkpoint_freq == 0:
                self.save_checkpoint("periodic", metrics, episode + 1)

            # Check and save best model
            if avg_safety_value > self.best_safety_value:
                self.best_safety_value = avg_safety_value
                metrics["best_safety_value"] = self.best_safety_value
                self.save_checkpoint("best", metrics, episode + 1)

            # Print progress
            if (episode + 1) % eval_freq == 0:
                print(f"Episode {episode + 1}/{n_episodes}")
                print(f"  Average Safety Value: {avg_safety_value:.3f}")
                print(f"  Best Safety Value: {self.best_safety_value:.3f}")
                print(f"  Collision Rate: {collision_rate:.3f}")
                print(f"  Average Loss: {metrics['losses'][-1]:.3f}")
                print(f"  Episode Length: {episode_length}")
                if torch.cuda.is_available():
                    print(
                        f"  GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB"
                    )
                print("----------------------------------------")

        return metrics


if __name__ == "__main__":
    # Enable CUDA benchmarking for optimized GPU operations
    torch.backends.cudnn.benchmark = True

    # Create environment instance
    from env import SafetyAwareRobotEnv

    env = SafetyAwareRobotEnv(render_mode="rgb_array")

    # Generate unique timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create structured checkpoint directory
    # Format: checkpoints/run_YYYYMMDD_HHMMSS/
    checkpoint_dir = os.path.join("checkpoints", f"run_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Log training initialization
    print(f"\nInitializing training run at: {timestamp}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Using device: {device}")

    try:
        # Initialize the SafetyLearner with specified hyperparameters
        learner = SafetyLearner(
            env=env,
            hidden_dim=256,
            gamma=0.999,
            learning_rate=3e-4,
            buffer_size=int(1e6),
            batch_size=64,
            action_limit=env.max_acc,
            checkpoint_dir=checkpoint_dir,
        )

        # Execute training loop with checkpointing configuration
        metrics = learner.train(
            n_episodes=10_000,
            max_steps=1000,
            checkpoint_freq=100,  # Save periodic checkpoint every 100 episodes
            eval_freq=10,  # Evaluate and print metrics every 10 episodes
        )

        # Prepare final checkpoint data
        final_checkpoint = {
            "model_state_dict": learner.q_network.state_dict(),
            "optimizer_state_dict": learner.optimizer.state_dict(),
            "metrics": metrics,
            "steps": learner.steps,
            "best_safety_value": learner.best_safety_value,
            "training_completed": True,
            "timestamp": timestamp,
        }

        # Save final model state
        final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
        torch.save(final_checkpoint, final_model_path)

        # Save comprehensive metrics record
        final_metrics_path = os.path.join(checkpoint_dir, "final_metrics.json")
        with open(final_metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # Save training configuration for reproducibility
        config = {
            "hidden_dim": 256,
            "gamma": 0.999,
            "learning_rate": 3e-4,
            "buffer_size": int(1e6),
            "batch_size": 64,
            "n_episodes": 10_000,
            "max_steps": 1000,
            "checkpoint_freq": 100,
            "eval_freq": 10,
            "device": str(device),
            "timestamp": timestamp,
        }
        config_path = os.path.join(checkpoint_dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        # Print training summary
        print("\nTraining completed successfully!")
        print(f"Total steps: {learner.steps}")
        print(f"Best safety value achieved: {learner.best_safety_value:.3f}")
        print(f"Final collision rate: {metrics['collision_rates'][-1]:.3f}")
        print(f"\nArtifacts saved in: {checkpoint_dir}")
        print(f"  - Final model: final_model.pth")
        print(f"  - Final metrics: final_metrics.json")
        print(f"  - Training config: training_config.json")

    except Exception as e:
        # Log any errors that occurred during training
        error_log_path = os.path.join(checkpoint_dir, "error_log.txt")
        with open(error_log_path, "w") as f:
            f.write(f"Error occurred during training: {str(e)}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        raise  # Re-raise the exception after logging
