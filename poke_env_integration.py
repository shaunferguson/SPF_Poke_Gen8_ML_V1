"""
Integration adapter for poke-env library
Bridges the DQN agent with poke-env's battle interface

To use this, install poke-env:
    pip install poke-env

Note: This is a template/example. You'll need to install poke-env and
adjust for your specific needs.
"""

try:
    from poke_env.player import Player, RandomPlayer
    from poke_env.environment import Battle, Move, Pokemon
    POKE_ENV_AVAILABLE = True
except ImportError:
    POKE_ENV_AVAILABLE = False
    print("poke-env not installed. Install with: pip install poke-env")

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional

from battle_agent import BattleAgent
from team_loader import TeamLoader


if POKE_ENV_AVAILABLE:
    class DQNPlayer(Player):
        """Player that uses DQN agent for decision making"""

        def __init__(self, battle_agent: BattleAgent, *args, **kwargs):
            """
            Initialize DQN player

            Args:
                battle_agent: The BattleAgent to use for decisions
            """
            super().__init__(*args, **kwargs)
            self.battle_agent = battle_agent
            self.previous_battle = None

        def choose_move(self, battle: Battle):
            """
            Choose move using DQN agent

            Args:
                battle: Current battle state from poke-env

            Returns:
                Move or switch choice
            """
            # Convert battle to our state format
            battle_state = self._convert_poke_env_battle(battle)

            # Calculate reward if we have a previous battle state
            if self.previous_battle is not None:
                prev_state = self._convert_poke_env_battle(self.previous_battle)
                reward = self.battle_agent.compute_reward(prev_state, battle_state)

                # Store transition
                done = battle.finished
                self.battle_agent.store_transition(battle_state, reward, done)

                # Train
                self.battle_agent.train_step()

            # Get action from agent
            action_str = self.battle_agent.select_action(battle_state, training=True)

            # Store for next iteration
            self.previous_battle = battle

            # Convert our action to poke-env format
            return self._convert_action_to_poke_env(action_str, battle)

        def _convert_poke_env_battle(self, battle: Battle) -> Dict[str, Any]:
            """
            Convert poke-env Battle object to our state format

            Args:
                battle: poke-env Battle object

            Returns:
                Battle state dictionary in our format
            """
            # Get active Pokemon
            active_pokemon = battle.active_pokemon
            opponent_active = battle.opponent_active_pokemon

            # Build team info
            team_pokemon = []
            for pokemon in battle.team.values():
                team_pokemon.append({
                    'species': pokemon.species,
                    'condition': f"{pokemon.current_hp}/{pokemon.max_hp}" if pokemon.max_hp else "0 fnt",
                    'active': pokemon == active_pokemon,
                })

            # Build opponent info (limited knowledge)
            opponent_pokemon = []
            for pokemon in battle.opponent_team.values():
                opponent_pokemon.append({
                    'species': pokemon.species,
                    'condition': f"{pokemon.current_hp}/{pokemon.max_hp}" if pokemon.max_hp else "0 fnt",
                    'active': pokemon == opponent_active,
                })

            # Get available moves
            available_moves = []
            if active_pokemon:
                for move_id, move in battle.available_moves:
                    available_moves.append({
                        'move': move.id,
                        'pp': move.current_pp,
                        'maxpp': move.max_pp,
                    })

            # Build state
            state = {
                'team': {
                    'active': {
                        'species': active_pokemon.species if active_pokemon else None,
                        'condition': f"{active_pokemon.current_hp}/{active_pokemon.max_hp}" if active_pokemon else "0 fnt",
                    } if active_pokemon else None,
                    'pokemon': team_pokemon,
                },
                'opponent': {
                    'active': {
                        'species': opponent_active.species if opponent_active else None,
                        'condition': f"{opponent_active.current_hp}/{opponent_active.max_hp}" if opponent_active else "0 fnt",
                    } if opponent_active else None,
                    'pokemon': opponent_pokemon,
                },
                'field': {
                    'weather': battle.weather.name.lower() if battle.weather else '',
                },
                'request': {
                    'active': [{
                        'moves': available_moves
                    }],
                    'side': {
                        'pokemon': team_pokemon
                    }
                },
                'winner': battle.winner if battle.finished else None,
                'player_id': 'p1',  # Simplified
            }

            return state

        def _convert_action_to_poke_env(self, action_str: str, battle: Battle):
            """
            Convert our action string to poke-env move

            Args:
                action_str: Our action string (e.g., "move 1", "switch 2")
                battle: Current battle

            Returns:
                poke-env move choice
            """
            parts = action_str.split()
            action_type = parts[0]
            action_idx = int(parts[1]) - 1  # Convert to 0-indexed

            if action_type == "move":
                # Select move by index
                available_moves = list(battle.available_moves)
                if action_idx < len(available_moves):
                    return self.create_order(available_moves[action_idx])
                else:
                    # Fallback to first available move
                    if available_moves:
                        return self.create_order(available_moves[0])

            elif action_type == "switch":
                # Select switch by index
                available_switches = list(battle.available_switches)
                if action_idx < len(available_switches):
                    return self.create_order(available_switches[action_idx])

            # Fallback: random valid move
            return self.choose_random_move(battle)

        def _battle_finished_callback(self, battle: Battle):
            """Called when battle finishes"""
            if self.previous_battle is not None:
                prev_state = self._convert_poke_env_battle(self.previous_battle)
                final_state = self._convert_poke_env_battle(battle)

                # Final reward
                reward = self.battle_agent.compute_reward(prev_state, final_state)
                self.battle_agent.store_transition(final_state, reward, done=True)

                # Final training step
                self.battle_agent.train_step()

            # Reset for next battle
            self.battle_agent.reset_episode()
            self.previous_battle = None


async def train_with_poke_env(num_battles: int = 1000):
    """
    Train agent using poke-env

    Args:
        num_battles: Number of battles to train
    """
    if not POKE_ENV_AVAILABLE:
        print("Cannot train: poke-env not installed")
        return

    # Load teams
    team_loader = TeamLoader()
    if team_loader.get_team_count() == 0:
        print("No teams loaded!")
        return

    # Create agent
    agent = BattleAgent()

    # Get a random team (poke-env format)
    team = team_loader.get_random_team()
    team_str = team_loader.export_team_for_showdown(team)

    # Create DQN player
    dqn_player = DQNPlayer(
        battle_agent=agent,
        battle_format="gen8ou",  # You'll want to use a custom format
        team=team_str,
    )

    # Create opponent (random player for now)
    opponent_team = team_loader.get_random_team()
    opponent_team_str = team_loader.export_team_for_showdown(opponent_team)

    opponent = RandomPlayer(
        battle_format="gen8ou",
        team=opponent_team_str,
    )

    # Train by battling
    await dqn_player.battle_against(opponent, n_battles=num_battles)

    # Save agent
    agent.save("checkpoints/agent_poke_env.pt")

    print(f"\nTraining complete!")
    print(f"Final stats: {agent.get_stats()}")


async def evaluate_with_poke_env(checkpoint_path: str, num_battles: int = 100):
    """
    Evaluate trained agent using poke-env

    Args:
        checkpoint_path: Path to agent checkpoint
        num_battles: Number of battles to evaluate
    """
    if not POKE_ENV_AVAILABLE:
        print("Cannot evaluate: poke-env not installed")
        return

    # Load teams
    team_loader = TeamLoader()

    # Load agent
    agent = BattleAgent()
    agent.load(checkpoint_path)

    # Create player
    team = team_loader.get_random_team()
    team_str = team_loader.export_team_for_showdown(team)

    dqn_player = DQNPlayer(
        battle_agent=agent,
        battle_format="gen8ou",
        team=team_str,
    )

    # Create opponent
    opponent_team = team_loader.get_random_team()
    opponent_team_str = team_loader.export_team_for_showdown(opponent_team)

    opponent = RandomPlayer(
        battle_format="gen8ou",
        team=opponent_team_str,
    )

    # Evaluate
    await dqn_player.battle_against(opponent, n_battles=num_battles)

    # Print results
    print(f"\nEvaluation Results:")
    print(f"Wins: {dqn_player.n_won_battles}")
    print(f"Losses: {dqn_player.n_lost_battles}")
    print(f"Win Rate: {dqn_player.n_won_battles / num_battles * 100:.1f}%")


def main():
    """Main entry point for poke-env integration"""
    if not POKE_ENV_AVAILABLE:
        print("\nThis script requires poke-env.")
        print("Install it with: pip install poke-env")
        print("\nThen you can use this to train your agent with actual battles!")
        return

    import argparse

    parser = argparse.ArgumentParser(description="Train with poke-env")
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate'],
                        help='Mode: train or evaluate')
    parser.add_argument('--battles', type=int, default=100,
                        help='Number of battles')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path for evaluation')

    args = parser.parse_args()

    if args.mode == 'train':
        asyncio.run(train_with_poke_env(num_battles=args.battles))
    elif args.mode == 'evaluate':
        if not args.checkpoint:
            print("Please specify --checkpoint for evaluation")
            return
        asyncio.run(evaluate_with_poke_env(args.checkpoint, num_battles=args.battles))


if __name__ == "__main__":
    main()
