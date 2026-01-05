"""
Training Script for Pokemon Battle Agent
Coordinates battles and training using DQN
"""

import asyncio
import argparse
import json
import os
from datetime import datetime
from typing import Optional

from showdown_client import ShowdownClient
from battle_agent import BattleAgent
from team_loader import TeamLoader


class BattleTrainer:
    """Coordinates training battles for the agent"""

    def __init__(self, agent: BattleAgent, team_loader: TeamLoader,
                 username: str = "PokemonBot", password: Optional[str] = None,
                 server_url: str = "ws://sim3.psim.us:8000/showdown/websocket",
                 save_dir: str = "checkpoints",
                 log_dir: str = "logs"):
        """
        Initialize battle trainer

        Args:
            agent: Battle agent to train
            team_loader: Team loader for getting teams
            username: Showdown username
            password: Showdown password (optional)
            server_url: Showdown server URL
            save_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.agent = agent
        self.team_loader = team_loader
        self.username = username
        self.password = password
        self.server_url = server_url
        self.save_dir = save_dir
        self.log_dir = log_dir

        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Battle tracking
        self.current_battle_id = None
        self.current_battle_state = None
        self.previous_battle_state = None
        self.battle_count = 0

        # Client
        self.client = None

    async def battle_callback(self, battle_id: str, battle_state: dict):
        """
        Callback for handling battle requests

        Args:
            battle_id: ID of the battle
            battle_state: Current battle state
        """
        self.current_battle_id = battle_id

        # Calculate reward if we have a previous state
        if self.previous_battle_state is not None:
            reward = self.agent.compute_reward(self.previous_battle_state, battle_state)

            # Check if battle ended
            done = battle_state.get('winner') is not None

            # Store transition
            self.agent.store_transition(battle_state, reward, done)

            # Train
            loss = self.agent.train_step()

            if loss is not None and self.battle_count % 10 == 0:
                stats = self.agent.get_stats()
                print(f"Battle {self.battle_count} | Loss: {loss:.4f} | "
                      f"Epsilon: {stats['epsilon']:.4f} | "
                      f"Avg Reward (100): {stats['avg_reward_100']:.2f}")

            # Reset if battle ended
            if done:
                self.battle_count += 1
                self.agent.reset_episode()
                self.previous_battle_state = None

                # Log battle result
                winner = battle_state.get('winner')
                result = "Win" if winner == self.username else "Loss"
                print(f"\nBattle {self.battle_count} finished: {result}")

                # Save checkpoint periodically
                if self.battle_count % 100 == 0:
                    self.save_checkpoint()

                # Start new battle after a delay
                await asyncio.sleep(2)
                await self.start_new_battle()
                return

        # Store current state as previous
        self.previous_battle_state = battle_state.copy()

        # Select and make action
        try:
            action = self.agent.select_action(battle_state, training=True)
            await self.client.make_move(battle_id, action)
        except Exception as e:
            print(f"Error selecting/making action: {e}")
            # Fallback to random move
            await self.client.make_move(battle_id, "move 1")

    async def start_new_battle(self):
        """Start a new battle with a random team"""
        # Get random team
        team = self.team_loader.get_random_team()
        if not team:
            print("No teams available!")
            return

        team_text = self.team_loader.export_team_for_showdown(team)

        # For now, we'll need to set up battles against another bot or use local server
        # This is a simplified version - in production you'd want to:
        # 1. Run a local Showdown server
        # 2. Connect two bots (or bot vs AI)
        # 3. Use a custom battle format for Gen 8 without Dynamax

        print(f"Starting battle {self.battle_count + 1} with team: {team['filename']}")

        # Note: For training, you'll want to either:
        # - Set up a local Showdown server and battle against a random/scripted opponent
        # - Use ladder battles (but this requires proper team building)
        # - Implement self-play (bot vs bot)

    def save_checkpoint(self):
        """Save training checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent_battle_{self.battle_count}_{timestamp}.pt"
        filepath = os.path.join(self.save_dir, filename)

        self.agent.save(filepath)

        # Also save latest
        latest_path = os.path.join(self.save_dir, "agent_latest.pt")
        self.agent.save(latest_path)

        # Save training log
        stats = self.agent.get_stats()
        log_data = {
            'timestamp': timestamp,
            'battle_count': self.battle_count,
            'stats': stats
        }

        log_path = os.path.join(self.log_dir, f"training_log_{timestamp}.json")
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

    async def train(self, num_battles: int = 1000):
        """
        Train the agent

        Args:
            num_battles: Number of battles to train for
        """
        print(f"Starting training for {num_battles} battles")
        print(f"Agent stats: {self.agent.get_stats()}")

        # Connect to Showdown
        self.client = ShowdownClient(
            username=self.username,
            password=self.password,
            server_url=self.server_url
        )

        await self.client.connect()
        await self.client.login()

        # Start first battle
        await self.start_new_battle()

        # Listen for battles
        await self.client.listen(battle_callback=self.battle_callback)


class SelfPlayTrainer:
    """Self-play trainer for agent vs agent battles"""

    def __init__(self, agent1: BattleAgent, agent2: Optional[BattleAgent] = None,
                 team_loader: TeamLoader = None,
                 save_dir: str = "checkpoints",
                 log_dir: str = "logs"):
        """
        Initialize self-play trainer

        Args:
            agent1: First agent (training agent)
            agent2: Second agent (opponent, if None uses copy of agent1)
            team_loader: Team loader
            save_dir: Checkpoint directory
            log_dir: Log directory
        """
        self.agent1 = agent1
        self.agent2 = agent2 if agent2 else BattleAgent()
        self.team_loader = team_loader
        self.save_dir = save_dir
        self.log_dir = log_dir

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    def simulate_battle(self, team1_text: str, team2_text: str) -> dict:
        """
        Simulate a battle between two teams

        Args:
            team1_text: Team text for agent 1
            team2_text: Team text for agent 2

        Returns:
            Battle result dictionary
        """
        # This would require implementing a battle simulator
        # For now, this is a placeholder

        # In a full implementation:
        # 1. Initialize battle state from teams
        # 2. Loop until battle ends:
        #    - Agent 1 selects action
        #    - Agent 2 selects action
        #    - Apply actions and update state
        #    - Give rewards
        #    - Train both agents
        # 3. Return result

        print("Battle simulation not yet implemented")
        print("You'll need to either:")
        print("1. Use Pokemon Showdown server for battles")
        print("2. Implement a battle engine (complex)")
        print("3. Use poke-env library for simulation")

        return {
            'winner': 1,
            'turns': 0
        }

    def train(self, num_battles: int = 1000):
        """
        Train using self-play

        Args:
            num_battles: Number of battles to simulate
        """
        print(f"Starting self-play training for {num_battles} battles")

        for battle_num in range(num_battles):
            # Get random teams
            team1 = self.team_loader.get_random_team()
            team2 = self.team_loader.get_random_team()

            if not team1 or not team2:
                print("Not enough teams!")
                break

            # Simulate battle
            result = self.simulate_battle(
                team1['team_text'],
                team2['team_text']
            )

            # Save checkpoint periodically
            if (battle_num + 1) % 100 == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(self.save_dir, f"agent1_battle_{battle_num+1}_{timestamp}.pt")
                self.agent1.save(filepath)

                stats = self.agent1.get_stats()
                print(f"Battle {battle_num + 1}/{num_battles} | Stats: {stats}")


def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(description="Train Pokemon battle agent")
    parser.add_argument('--mode', type=str, default='selfplay',
                        choices=['online', 'selfplay'],
                        help='Training mode (online or selfplay)')
    parser.add_argument('--battles', type=int, default=1000,
                        help='Number of battles to train for')
    parser.add_argument('--username', type=str, default='PokemonBot',
                        help='Showdown username')
    parser.add_argument('--password', type=str, default=None,
                        help='Showdown password')
    parser.add_argument('--teams-dir', type=str, default='bdsp_BT_teams',
                        help='Directory containing team files')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Load agent from checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Load teams
    print(f"Loading teams from {args.teams_dir}...")
    team_loader = TeamLoader(args.teams_dir)

    if team_loader.get_team_count() == 0:
        print("No teams loaded! Please check your teams directory.")
        return

    # Create agent
    print("Initializing agent...")
    agent = BattleAgent(device=args.device)

    # Load checkpoint if specified
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        agent.load(args.checkpoint)

    # Train
    if args.mode == 'online':
        print("Starting online training...")
        trainer = BattleTrainer(
            agent=agent,
            team_loader=team_loader,
            username=args.username,
            password=args.password
        )
        asyncio.run(trainer.train(num_battles=args.battles))

    elif args.mode == 'selfplay':
        print("Starting self-play training...")
        print("\nNOTE: Self-play mode requires a battle simulator.")
        print("Consider using poke-env library or connecting to Showdown server.")
        print("For now, this will create the framework but battles won't run.\n")

        trainer = SelfPlayTrainer(
            agent1=agent,
            team_loader=team_loader
        )
        trainer.train(num_battles=args.battles)


if __name__ == "__main__":
    main()
