import torch
import numpy as np
from env import SafetyAwareRobotEnv
from train import SafetyQNetwork, DiscreteActionWrapper
import matplotlib.pyplot as plt
from typing import List, Tuple
import os


def load_safety_model(model_path: str, env: SafetyAwareRobotEnv) -> SafetyQNetwork:
    """
    Load the trained safety network with proper error handling.

    Args:
        model_path: Path to the saved model weights
        env: Environment instance for determining network dimensions

    Returns:
        Loaded SafetyQNetwork instance
    """
    # Initialize network with same dimensions as training
    state_dim = env.observation_space.shape[0]
    action_dim = DiscreteActionWrapper(env, n_bins=5).action_space.n
    network = SafetyQNetwork(state_dim, action_dim)

    # Load weights with error handling
    try:
        # Use weights_only=True for safer model loading
        network.load_state_dict(torch.load(model_path, weights_only=True))
        network.eval()  # Set to evaluation mode
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"No model file found at {model_path}")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

    return network


def evaluate_episode(
    env: DiscreteActionWrapper,  # Note: Now expects wrapped environment
    network: SafetyQNetwork,
    max_steps: int = 1000,
    render: bool = True,
) -> Tuple[List[float], List[np.ndarray], bool]:
    """
    Run a single evaluation episode and collect metrics.

    Args:
        env: Wrapped environment instance with discrete action space
        network: Loaded safety network
        max_steps: Maximum steps per episode
        render: Whether to render frames

    Returns:
        Tuple of (safety_values, trajectory, collision_occurred)
    """
    state, _ = env.reset()
    safety_values = []
    trajectory = []
    collision_occurred = False

    for step in range(max_steps):
        # Get safety values for all actions
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            safety_predictions = network(state_tensor)

        # Select action with highest safety value
        discrete_action = safety_predictions.argmax().item()

        # Record safety value and state
        safety_values.append(safety_predictions.max().item())
        trajectory.append(state.copy())

        # Execute action in wrapped environment
        state, _, terminated, truncated, _ = env.step(discrete_action)

        # Render if requested
        if render:
            env.render()

        # Check termination
        if terminated:
            collision_occurred = True
            break
        if truncated:
            break

    return safety_values, trajectory, collision_occurred


def visualize_results(
    safety_values: List[float], trajectory: List[np.ndarray], env: DiscreteActionWrapper
):
    """
    Create visualization of the evaluation results.

    Args:
        safety_values: List of safety values during episode
        trajectory: List of states visited
        env: Wrapped environment instance
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot safety values over time
    ax1.plot(safety_values)
    ax1.set_title("Safety Values During Episode")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Predicted Safety Value")
    ax1.grid(True)

    # Plot trajectory and obstacles
    trajectory = np.array(trajectory)
    ax2.plot(trajectory[:, 0], trajectory[:, 1], "b-", label="Robot Path")

    # Plot obstacles - access base environment's obstacles
    for obs_x, obs_y, obs_r in env.env.obstacles:  # Note: env.env to access base env
        circle = plt.Circle((obs_x, obs_y), obs_r, color="gray", alpha=0.5)
        ax2.add_artist(circle)

    # Plot start and end positions
    ax2.plot(trajectory[0, 0], trajectory[0, 1], "go", label="Start")
    ax2.plot(trajectory[-1, 0], trajectory[-1, 1], "ro", label="End")

    ax2.set_title("Robot Trajectory")
    ax2.set_xlabel("X Position")
    ax2.set_ylabel("Y Position")
    ax2.grid(True)
    ax2.legend()
    ax2.set_aspect("equal")

    # Set axis limits based on environment bounds
    max_pos = env.env.max_pos
    ax2.set_xlim(-max_pos, max_pos)
    ax2.set_ylim(-max_pos, max_pos)

    plt.tight_layout()
    plt.show()


def main():
    # Create base environment
    base_env = SafetyAwareRobotEnv(render_mode="human")

    # Wrap with discrete actions - CRITICAL STEP!
    env = DiscreteActionWrapper(base_env, n_bins=5)

    # Load model
    model_path = "safety_network.pth"
    try:
        network = load_safety_model(model_path, env)
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return

    # Run evaluation episode
    safety_values, trajectory, collision_occurred = evaluate_episode(
        env, network, max_steps=1000, render=True
    )

    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Episode Length: {len(safety_values)}")
    print(f"Average Safety Value: {np.mean(safety_values):.3f}")
    print(f"Collision Occurred: {collision_occurred}")

    # Visualize results
    visualize_results(safety_values, trajectory, env)

    env.close()


if __name__ == "__main__":
    main()
