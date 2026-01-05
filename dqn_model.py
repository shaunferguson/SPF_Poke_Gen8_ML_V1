"""
Deep Q-Network (DQN) Model for Pokemon Battling
Implements the neural network architecture for Q-learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random
from typing import List, Tuple


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'action_mask', 'next_action_mask'])


class DQN(nn.Module):
    """Deep Q-Network for Pokemon battle decision making"""

    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = None):
        """
        Initialize DQN

        Args:
            state_size: Dimension of state vector
            action_size: Number of possible actions
            hidden_sizes: List of hidden layer sizes (default: [512, 256, 128])
        """
        super(DQN, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]

        self.state_size = state_size
        self.action_size = action_size

        # Build network layers
        layers = []
        input_size = state_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_size = hidden_size

        # Output layer
        layers.append(nn.Linear(input_size, action_size))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network

        Args:
            state: State tensor

        Returns:
            Q-values for each action
        """
        return self.network(state)


class DuelingDQN(nn.Module):
    """Dueling DQN architecture for improved performance"""

    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = None):
        """
        Initialize Dueling DQN

        Args:
            state_size: Dimension of state vector
            action_size: Number of possible actions
            hidden_sizes: List of hidden layer sizes
        """
        super(DuelingDQN, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [512, 256]

        self.state_size = state_size
        self.action_size = action_size

        # Shared feature layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_sizes[1], 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_sizes[1], 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dueling network

        Args:
            state: State tensor

        Returns:
            Q-values for each action
        """
        features = self.feature_layer(state)

        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine value and advantages: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))

        return q_values


class ReplayBuffer:
    """Experience replay buffer for DQN training"""

    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer

        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state: torch.Tensor, action: int, reward: float,
             next_state: torch.Tensor, done: bool,
             action_mask: torch.Tensor, next_action_mask: torch.Tensor):
        """
        Add experience to buffer

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            action_mask: Mask of legal actions in current state
            next_action_mask: Mask of legal actions in next state
        """
        experience = Experience(
            state.cpu(),
            action,
            reward,
            next_state.cpu(),
            done,
            action_mask.cpu(),
            next_action_mask.cpu()
        )
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample batch of experiences

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Batch of states, actions, rewards, next_states, dones, masks
        """
        experiences = random.sample(self.buffer, batch_size)

        states = torch.stack([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.stack([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        action_masks = torch.stack([e.action_mask for e in experiences])
        next_action_masks = torch.stack([e.next_action_mask for e in experiences])

        return states, actions, rewards, next_states, dones, action_masks, next_action_masks

    def __len__(self) -> int:
        """Return current size of buffer"""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer"""

    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        """
        Initialize prioritized replay buffer

        Args:
            capacity: Maximum number of experiences
            alpha: Priority exponent (how much prioritization to use)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, state: torch.Tensor, action: int, reward: float,
             next_state: torch.Tensor, done: bool,
             action_mask: torch.Tensor, next_action_mask: torch.Tensor):
        """Add experience with maximum priority"""
        max_priority = max(self.priorities) if self.priorities else 1.0

        experience = Experience(
            state.cpu(),
            action,
            reward,
            next_state.cpu(),
            done,
            action_mask.cpu(),
            next_action_mask.cpu()
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[torch.Tensor, ...]:
        """
        Sample batch with prioritization

        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling exponent

        Returns:
            Batch of experiences plus importance sampling weights and indices
        """
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]

        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights)

        states = torch.stack([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.stack([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        action_masks = torch.stack([e.action_mask for e in experiences])
        next_action_masks = torch.stack([e.next_action_mask for e in experiences])

        return states, actions, rewards, next_states, dones, action_masks, next_action_masks, weights, indices

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self) -> int:
        """Return current size of buffer"""
        return len(self.buffer)


class EpsilonGreedyPolicy:
    """Epsilon-greedy exploration policy"""

    def __init__(self, epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995):
        """
        Initialize epsilon-greedy policy

        Args:
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate per step
        """
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def select_action(self, q_values: torch.Tensor, action_mask: torch.Tensor) -> int:
        """
        Select action using epsilon-greedy policy

        Args:
            q_values: Q-values for each action
            action_mask: Mask of legal actions

        Returns:
            Selected action index
        """
        # Get legal actions
        legal_actions = action_mask.nonzero(as_tuple=True)[0]

        if len(legal_actions) == 0:
            # No legal actions (shouldn't happen, but handle it)
            return 0

        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            # Explore: random legal action
            return legal_actions[random.randint(0, len(legal_actions) - 1)].item()
        else:
            # Exploit: best legal action
            masked_q_values = q_values.clone()
            masked_q_values[action_mask == 0] = float('-inf')
            return masked_q_values.argmax().item()

    def decay_epsilon(self):
        """Decay epsilon value"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def compute_loss(policy_net: nn.Module, target_net: nn.Module,
                 states: torch.Tensor, actions: torch.Tensor,
                 rewards: torch.Tensor, next_states: torch.Tensor,
                 dones: torch.Tensor, action_masks: torch.Tensor,
                 next_action_masks: torch.Tensor,
                 gamma: float = 0.99) -> torch.Tensor:
    """
    Compute DQN loss (Double DQN variant)

    Args:
        policy_net: Policy network
        target_net: Target network
        states: Batch of states
        actions: Batch of actions
        rewards: Batch of rewards
        next_states: Batch of next states
        dones: Batch of done flags
        action_masks: Batch of action masks
        next_action_masks: Batch of next action masks
        gamma: Discount factor

    Returns:
        Loss tensor
    """
    # Get current Q-values
    current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute next Q-values using Double DQN
    with torch.no_grad():
        # Use policy net to select actions
        next_q_values_policy = policy_net(next_states)
        next_q_values_policy[next_action_masks == 0] = float('-inf')
        next_actions = next_q_values_policy.argmax(1)

        # Use target net to evaluate actions
        next_q_values_target = target_net(next_states)
        next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

    # Compute loss (Huber loss for stability)
    loss = F.smooth_l1_loss(current_q_values, target_q_values)

    return loss


def test_model():
    """Test the DQN model"""
    state_size = 200
    action_size = 9

    # Test standard DQN
    model = DQN(state_size, action_size)
    sample_state = torch.randn(1, state_size)
    q_values = model(sample_state)
    print(f"DQN output shape: {q_values.shape}")

    # Test Dueling DQN
    dueling_model = DuelingDQN(state_size, action_size)
    q_values_dueling = dueling_model(sample_state)
    print(f"Dueling DQN output shape: {q_values_dueling.shape}")

    # Test epsilon-greedy policy
    policy = EpsilonGreedyPolicy()
    action_mask = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.float32)
    action = policy.select_action(q_values[0], action_mask)
    print(f"Selected action: {action}")

    # Test replay buffer
    buffer = ReplayBuffer(capacity=1000)
    next_state = torch.randn(1, state_size)
    next_mask = torch.tensor([1, 1, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float32)

    buffer.push(sample_state[0], action, 1.0, next_state[0], False, action_mask, next_mask)
    print(f"Buffer size: {len(buffer)}")


if __name__ == "__main__":
    test_model()
