"""
Battle Agent for Pokemon Showdown
Coordinates DQN model with game state to make battle decisions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Tuple
import numpy as np
import os
import json
from datetime import datetime

from state_encoder import BattleStateEncoder
from dqn_model import DQN, DuelingDQN, ReplayBuffer, PrioritizedReplayBuffer, EpsilonGreedyPolicy, compute_loss


class BattleAgent:
    """Agent that learns to battle using DQN"""

    def __init__(self, model_type: str = 'dueling', use_prioritized_replay: bool = True,
                 learning_rate: float = 0.0001, gamma: float = 0.99,
                 batch_size: int = 64, buffer_capacity: int = 100000,
                 target_update_frequency: int = 1000,
                 device: str = None):
        """
        Initialize battle agent

        Args:
            model_type: Type of model ('dqn' or 'dueling')
            use_prioritized_replay: Whether to use prioritized experience replay
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            batch_size: Batch size for training
            buffer_capacity: Replay buffer capacity
            target_update_frequency: Steps between target network updates
            device: Device to use ('cuda' or 'cpu')
        """
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Initialize encoder
        self.encoder = BattleStateEncoder()
        self.state_size = self.encoder.state_size
        self.action_size = 4 + 5  # 4 moves + 5 possible switches (6 Pokemon - 1 active)

        # Initialize networks
        if model_type == 'dueling':
            self.policy_net = DuelingDQN(self.state_size, self.action_size).to(self.device)
            self.target_net = DuelingDQN(self.state_size, self.action_size).to(self.device)
        else:
            self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
            self.target_net = DQN(self.state_size, self.action_size).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Initialize replay buffer
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)
        else:
            self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.use_prioritized_replay = use_prioritized_replay

        # Exploration policy
        self.policy = EpsilonGreedyPolicy()

        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

        # Training state
        self.steps = 0
        self.episodes = 0
        self.current_episode_reward = 0
        self.episode_rewards = []

        # Battle state tracking
        self.last_state = None
        self.last_action = None
        self.last_action_mask = None

    def select_action(self, battle_state: Dict[str, Any], training: bool = True) -> str:
        """
        Select action based on current battle state

        Args:
            battle_state: Current battle state from ShowdownClient
            training: Whether in training mode (uses exploration)

        Returns:
            Action string (e.g., 'move 1', 'switch 2')
        """
        # Encode state
        state = self.encoder.encode_state(battle_state).to(self.device)
        action_mask = self.encoder.get_action_mask(battle_state).to(self.device)

        # Get Q-values
        with torch.no_grad():
            q_values = self.policy_net(state.unsqueeze(0))[0]

        # Select action
        if training:
            action_idx = self.policy.select_action(q_values, action_mask)
        else:
            # Greedy selection during evaluation
            masked_q_values = q_values.clone()
            masked_q_values[action_mask == 0] = float('-inf')
            action_idx = masked_q_values.argmax().item()

        # Store for learning
        self.last_state = state
        self.last_action = action_idx
        self.last_action_mask = action_mask

        # Convert action index to action string
        return self._action_to_string(action_idx, battle_state)

    def _action_to_string(self, action_idx: int, battle_state: Dict[str, Any]) -> str:
        """
        Convert action index to Showdown action string

        Args:
            action_idx: Action index (0-3 for moves, 4-8 for switches)
            battle_state: Current battle state

        Returns:
            Action string for Showdown server
        """
        if action_idx < 4:
            # Move action
            return f"move {action_idx + 1}"
        else:
            # Switch action
            switch_idx = action_idx - 4
            # Find the non-active Pokemon at this index
            request = battle_state.get('request', {})
            side = request.get('side', {})
            pokemon = side.get('pokemon', [])

            available_switches = []
            for i, mon in enumerate(pokemon):
                if not mon.get('active', False) and mon.get('condition', '').split()[0] != '0':
                    available_switches.append(i + 1)

            if switch_idx < len(available_switches):
                return f"switch {available_switches[switch_idx]}"
            else:
                # Fallback to first available move
                return "move 1"

    def store_transition(self, next_battle_state: Dict[str, Any], reward: float, done: bool):
        """
        Store transition in replay buffer

        Args:
            next_battle_state: Next battle state
            reward: Reward received
            done: Whether episode is done
        """
        if self.last_state is None:
            return

        # Encode next state
        next_state = self.encoder.encode_state(next_battle_state).to(self.device)
        next_action_mask = self.encoder.get_action_mask(next_battle_state).to(self.device)

        # Store in replay buffer
        self.replay_buffer.push(
            self.last_state,
            self.last_action,
            reward,
            next_state,
            done,
            self.last_action_mask,
            next_action_mask
        )

        # Update episode reward
        self.current_episode_reward += reward

        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
            self.episodes += 1

    def train_step(self) -> Optional[float]:
        """
        Perform one training step

        Returns:
            Loss value if training occurred, None otherwise
        """
        # Check if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        if self.use_prioritized_replay:
            beta = min(1.0, 0.4 + self.steps * (1.0 - 0.4) / 100000)
            batch = self.replay_buffer.sample(self.batch_size, beta)
            states, actions, rewards, next_states, dones, action_masks, next_action_masks, weights, indices = batch
            weights = weights.to(self.device)
        else:
            states, actions, rewards, next_states, dones, action_masks, next_action_masks = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)

        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        action_masks = action_masks.to(self.device)
        next_action_masks = next_action_masks.to(self.device)

        # Compute loss
        loss = compute_loss(
            self.policy_net, self.target_net,
            states, actions, rewards, next_states, dones,
            action_masks, next_action_masks,
            gamma=self.gamma
        )

        # Apply importance sampling weights for prioritized replay
        if self.use_prioritized_replay:
            loss = (loss * weights).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities if using prioritized replay
        if self.use_prioritized_replay:
            with torch.no_grad():
                current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q = self.target_net(next_states).max(1)[0]
                target_q = rewards + self.gamma * next_q * (1 - dones)
                td_errors = torch.abs(current_q - target_q).cpu().numpy()
                self.replay_buffer.update_priorities(indices, td_errors + 1e-6)

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.policy.decay_epsilon()

        return loss.item()

    def compute_reward(self, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> float:
        """
        Compute reward based on state transition

        Args:
            old_state: Previous battle state
            new_state: Current battle state

        Returns:
            Reward value
        """
        reward = 0.0

        # Win/loss rewards
        winner = new_state.get('winner')
        if winner:
            if winner == new_state.get('player_id'):
                reward += 100.0  # Win
            else:
                reward -= 100.0  # Loss
            return reward

        # HP-based rewards
        old_player_hp = self._get_team_hp_percentage(old_state.get('team', {}))
        new_player_hp = self._get_team_hp_percentage(new_state.get('team', {}))
        old_opponent_hp = self._get_team_hp_percentage(old_state.get('opponent', {}))
        new_opponent_hp = self._get_team_hp_percentage(new_state.get('opponent', {}))

        # Reward for dealing damage
        opponent_damage = old_opponent_hp - new_opponent_hp
        reward += opponent_damage * 10.0

        # Penalty for taking damage
        player_damage = old_player_hp - new_player_hp
        reward -= player_damage * 10.0

        # Penalty for fainting
        old_player_fainted = self._count_fainted(old_state.get('team', {}))
        new_player_fainted = self._count_fainted(new_state.get('team', {}))
        if new_player_fainted > old_player_fainted:
            reward -= 20.0

        # Reward for fainting opponent's Pokemon
        old_opponent_fainted = self._count_fainted(old_state.get('opponent', {}))
        new_opponent_fainted = self._count_fainted(new_state.get('opponent', {}))
        if new_opponent_fainted > old_opponent_fainted:
            reward += 20.0

        return reward

    @staticmethod
    def _get_team_hp_percentage(team: Dict[str, Any]) -> float:
        """Get average HP percentage of team"""
        pokemon = team.get('pokemon', [])
        if not pokemon:
            return 0.0

        total_hp = 0.0
        for mon in pokemon:
            condition = mon.get('condition', '100/100')
            try:
                if condition == '0 fnt':
                    hp_pct = 0.0
                else:
                    hp_part = condition.split()[0]
                    if '/' in hp_part:
                        current, max_hp = hp_part.split('/')
                        hp_pct = float(current) / float(max_hp)
                    else:
                        hp_pct = 1.0
                total_hp += hp_pct
            except (ValueError, ZeroDivisionError):
                total_hp += 1.0

        return total_hp / len(pokemon)

    @staticmethod
    def _count_fainted(team: Dict[str, Any]) -> int:
        """Count fainted Pokemon in team"""
        pokemon = team.get('pokemon', [])
        count = 0
        for mon in pokemon:
            condition = mon.get('condition', '100/100')
            if condition == '0 fnt' or condition.startswith('0 '):
                count += 1
        return count

    def save(self, path: str):
        """Save agent state"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.policy.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'episode_rewards': self.episode_rewards,
        }

        torch.save(checkpoint, path)
        print(f"Agent saved to {path}")

    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        self.episode_rewards = checkpoint['episode_rewards']

        print(f"Agent loaded from {path}")

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else [0]

        return {
            'episodes': self.episodes,
            'steps': self.steps,
            'epsilon': self.policy.epsilon,
            'avg_reward_100': np.mean(recent_rewards),
            'max_reward': max(self.episode_rewards) if self.episode_rewards else 0,
            'min_reward': min(self.episode_rewards) if self.episode_rewards else 0,
        }

    def reset_episode(self):
        """Reset episode-specific state"""
        self.last_state = None
        self.last_action = None
        self.last_action_mask = None


def test_agent():
    """Test the battle agent"""
    agent = BattleAgent()

    sample_state = {
        'team': {
            'active': {'species': 'Garchomp', 'condition': '75/100'},
            'pokemon': [
                {'species': 'Garchomp', 'condition': '75/100'},
                {'species': 'Ninetales', 'condition': '100/100'},
            ]
        },
        'opponent': {
            'active': {'species': 'Charizard', 'condition': '50/100'},
            'pokemon': [
                {'species': 'Charizard', 'condition': '50/100'},
            ]
        },
        'field': {},
        'request': {
            'active': [{
                'moves': [
                    {'move': 'Earthquake', 'pp': 10, 'maxpp': 10},
                    {'move': 'Dragon Claw', 'pp': 15, 'maxpp': 15},
                ]
            }],
            'side': {
                'pokemon': [
                    {'active': True, 'condition': '75/100'},
                    {'active': False, 'condition': '100/100'},
                ]
            }
        }
    }

    # Test action selection
    action = agent.select_action(sample_state)
    print(f"Selected action: {action}")

    # Test training step
    next_state = sample_state.copy()
    agent.store_transition(next_state, 1.0, False)

    # Get stats
    stats = agent.get_stats()
    print(f"Agent stats: {stats}")


if __name__ == "__main__":
    test_agent()
