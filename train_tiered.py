"""
Tier-Aware Training Script for Pokemon Battle Agent
Ensures battles only happen between teams of the same tier and format
"""

import asyncio
import argparse
import json
import os
import random
from datetime import datetime
from typing import Optional

from showdown_client import ShowdownClient
from battle_agent import BattleAgent
from team_loader_tiered import TieredTeamLoader


class TieredBattleTrainer:
    """Coordinates tier-aware training battles using local Showdown server"""

    def __init__(self, agent: BattleAgent, team_loader: TieredTeamLoader,
                 bot_username: str = "TrainingBot",
                 opponent_username: str = "OpponentBot",
                 server_url: str = "ws://127.0.0.1:8000/showdown/websocket",
                 tier: Optional[int] = None,
                 format_type: Optional[str] = None,
                 save_dir: str = "checkpoints",
                 log_dir: str = "logs"):
        """
        Initialize tiered battle trainer

        Args:
            agent: Battle agent to train
            team_loader: Tiered team loader
            bot_username: Training bot username
            opponent_username: Opponent bot username
            server_url: Local Showdown server URL
            tier: Specific tier (1-7) or None for all tiers
            format_type: 'S' for Singles, 'D' for Doubles, or None for both
            save_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.agent = agent
        self.team_loader = team_loader
        self.bot_username = bot_username
        self.opponent_username = opponent_username
        self.server_url = server_url
        self.preferred_tier = tier
        self.preferred_format = format_type
        self.save_dir = save_dir
        self.log_dir = log_dir

        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Battle tracking
        self.current_battle_id = None
        self.previous_battle_state = None
        self.battle_count = 0
        self.wins = 0
        self.losses = 0

        # Tier-specific tracking
        self.tier_stats = {i: {'wins': 0, 'losses': 0} for i in range(1, 8)}
        self.format_stats = {'S': {'wins': 0, 'losses': 0},
                            'D': {'wins': 0, 'losses': 0}}
        self.current_tier = None
        self.current_format = None

        # Clients
        self.bot_client = None
        self.opponent_client = None

    async def bot_battle_callback(self, battle_id: str, battle_state: dict):
        """Callback for training bot's battle decisions"""
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
                print(f"[Battle {self.battle_count}] Loss: {loss:.4f} | "
                      f"Epsilon: {stats['epsilon']:.4f} | "
                      f"Avg Reward (100): {stats['avg_reward_100']:.2f}")

            # Handle battle end
            if done:
                self.battle_count += 1
                winner = battle_state.get('winner')

                if winner == self.bot_client.username:
                    self.wins += 1
                    result = "WIN"

                    # Update tier-specific stats
                    if self.current_tier:
                        self.tier_stats[self.current_tier]['wins'] += 1
                    if self.current_format:
                        self.format_stats[self.current_format]['wins'] += 1
                else:
                    self.losses += 1
                    result = "LOSS"

                    # Update tier-specific stats
                    if self.current_tier:
                        self.tier_stats[self.current_tier]['losses'] += 1
                    if self.current_format:
                        self.format_stats[self.current_format]['losses'] += 1

                win_rate = self.wins / self.battle_count if self.battle_count > 0 else 0

                format_name = 'Singles' if self.current_format == 'S' else 'Doubles'
                print(f"\n[Battle {self.battle_count}] {result} | "
                      f"Tier {self.current_tier} {format_name} | "
                      f"W/L: {self.wins}/{self.losses} ({win_rate*100:.1f}%)")

                # Save checkpoint periodically
                if self.battle_count % 50 == 0:
                    self.save_checkpoint()
                    self.save_stats()
                    self.print_tier_stats()

                # Reset for next battle
                self.agent.reset_episode()
                self.previous_battle_state = None

                # Start new battle after delay
                await asyncio.sleep(1)
                await self.start_new_battle()
                return

        # Store current state
        self.previous_battle_state = battle_state.copy()

        # Select and make action
        try:
            action = self.agent.select_action(battle_state, training=True)
            await self.bot_client.make_move(battle_id, action)
        except Exception as e:
            print(f"Error in bot action: {e}")
            # Fallback
            await self.bot_client.make_move(battle_id, "move 1")

    async def opponent_battle_callback(self, battle_id: str, battle_state: dict):
        """Callback for opponent bot's battle decisions (random/simple AI)"""
        # Simple random opponent for now
        try:
            # Get request
            request = battle_state.get('request', {})
            if not request:
                return

            # Random move selection
            active_data = request.get('active', [{}])[0] if request.get('active') else {}
            moves = active_data.get('moves', [])

            # Filter available moves
            available_moves = [i for i, move in enumerate(moves)
                             if not move.get('disabled', False) and move.get('pp', 0) > 0]

            if available_moves:
                move_idx = random.choice(available_moves)
                await self.opponent_client.make_move(battle_id, f"move {move_idx + 1}")
            else:
                # Switch if no moves available
                await self.opponent_client.make_move(battle_id, "switch 2")

        except Exception as e:
            print(f"Error in opponent action: {e}")

    async def start_new_battle(self):
        """Start a new battle with matched tier and format"""
        # Get matching team pair (same tier and format)
        team_pair = self.team_loader.get_matching_team_pair(
            tier=self.preferred_tier,
            format_type=self.preferred_format
        )

        if not team_pair:
            print("Error: Could not find matching team pair")
            return

        bot_team, opponent_team = team_pair

        # Store current battle info
        self.current_tier = bot_team['tier']
        self.current_format = bot_team['format']

        bot_team_text = self.team_loader.export_team_for_showdown(bot_team)
        opponent_team_text = self.team_loader.export_team_for_showdown(opponent_team)

        format_name = 'Singles' if self.current_format == 'S' else 'Doubles'

        print(f"\n[Starting Battle {self.battle_count + 1}]")
        print(f"  Tier: {self.current_tier} | Format: {format_name}")
        print(f"  Bot: {bot_team['trainer_name']} ({bot_team['filename']})")
        print(f"  Opponent: {opponent_team['trainer_name']} ({opponent_team['filename']})")

        # Determine showdown format based on tier and format
        if self.current_format == 'S':
            battle_format = "gen8bdspssingles"
        else:
            battle_format = "gen8bdspbattletower"

        # Challenge opponent (use actual username from client, not the requested one)
        try:
            await self.bot_client.challenge_user(
                self.opponent_client.username,  # Use actual assigned username
                bot_team_text,
                battle_format=battle_format
            )
        except Exception as e:
            print(f"Error starting battle: {e}")

    def print_tier_stats(self):
        """Print statistics by tier"""
        print(f"\n{'='*60}")
        print("Performance by Tier")
        print('='*60)

        for tier in range(1, 8):
            stats = self.tier_stats[tier]
            total = stats['wins'] + stats['losses']
            if total > 0:
                win_rate = stats['wins'] / total * 100
                print(f"Tier {tier}: {stats['wins']:3d}W / {stats['losses']:3d}L ({win_rate:5.1f}%) - {total:3d} battles")

        print(f"\n{'='*60}")
        print("Performance by Format")
        print('='*60)

        for format_type, format_name in [('S', 'Singles'), ('D', 'Doubles')]:
            stats = self.format_stats[format_type]
            total = stats['wins'] + stats['losses']
            if total > 0:
                win_rate = stats['wins'] / total * 100
                print(f"{format_name}: {stats['wins']:3d}W / {stats['losses']:3d}L ({win_rate:5.1f}%) - {total:3d} battles")

        print('='*60)

    def save_checkpoint(self):
        """Save training checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent_battle_{self.battle_count}_{timestamp}.pt"
        filepath = os.path.join(self.save_dir, filename)

        self.agent.save(filepath)

        # Also save latest
        latest_path = os.path.join(self.save_dir, "agent_latest.pt")
        self.agent.save(latest_path)

    def save_stats(self):
        """Save training statistics including tier breakdown"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        stats = self.agent.get_stats()
        stats['battles'] = self.battle_count
        stats['wins'] = self.wins
        stats['losses'] = self.losses
        stats['win_rate'] = self.wins / self.battle_count if self.battle_count > 0 else 0
        stats['tier_stats'] = self.tier_stats
        stats['format_stats'] = self.format_stats

        log_path = os.path.join(self.log_dir, f"training_log_{timestamp}.json")
        with open(log_path, 'w') as f:
            json.dump(stats, f, indent=2)

        # Also save to CSV for easy tracking
        csv_path = os.path.join(self.log_dir, "training_history.csv")
        file_exists = os.path.exists(csv_path)

        with open(csv_path, 'a') as f:
            if not file_exists:
                f.write("timestamp,battles,wins,losses,win_rate,epsilon,avg_reward_100\n")
            f.write(f"{timestamp},{self.battle_count},{self.wins},{self.losses},"
                   f"{stats['win_rate']:.4f},{stats['epsilon']:.4f},{stats['avg_reward_100']:.4f}\n")

    async def train(self, num_battles: int = 1000):
        """
        Train the agent with tier-aware matchmaking

        Args:
            num_battles: Number of battles to train for
        """
        print("="*60)
        print("BDSP Battle AI - Tier-Aware Training")
        print("="*60)
        print(f"Target battles: {num_battles}")
        print(f"Agent stats: {self.agent.get_stats()}")
        print(f"Server: {self.server_url}")

        if self.preferred_tier:
            print(f"Tier: {self.preferred_tier} only")
        else:
            print("Tier: All tiers (random selection)")

        if self.preferred_format:
            format_name = 'Singles' if self.preferred_format == 'S' else 'Doubles'
            print(f"Format: {format_name} only")
        else:
            print("Format: Both Singles and Doubles")

        print("="*60)

        # Create bot client
        self.bot_client = ShowdownClient(
            username=self.bot_username,
            password=None,
            server_url=self.server_url
        )

        # Create opponent client
        self.opponent_client = ShowdownClient(
            username=self.opponent_username,
            password=None,
            server_url=self.server_url
        )

        # Connect both clients
        print("\nConnecting to local server...")
        await self.bot_client.connect()
        await self.bot_client.login()
        print(f"[OK] {self.bot_username} connected")

        await self.opponent_client.connect()
        await self.opponent_client.login()
        print(f"[OK] {self.opponent_username} connected")

        # Set up opponent to accept challenges
        async def opponent_listener():
            await self.opponent_client.listen(battle_callback=self.opponent_battle_callback)

        # Start opponent listener in background
        opponent_task = asyncio.create_task(opponent_listener())

        # Give opponent time to set up
        await asyncio.sleep(2)

        # Start first battle
        await self.start_new_battle()

        # Start bot listener
        await self.bot_client.listen(battle_callback=self.bot_battle_callback)


def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(description="Train with tier-aware matchmaking")
    parser.add_argument('--battles', type=int, default=1000,
                        help='Number of battles to train for')
    parser.add_argument('--tier', type=int, choices=range(1, 8), default=None,
                        help='Specific tier to train on (1-7), or omit for all tiers')
    parser.add_argument('--format', type=str, choices=['S', 'D', 'singles', 'doubles'],
                        default=None,
                        help='Format: S/singles or D/doubles, or omit for both')
    parser.add_argument('--teams-dir', type=str, default='bdsp_BT_teams',
                        help='Directory containing team files')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Load agent from checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--server', type=str, default='ws://127.0.0.1:8000/showdown/websocket',
                        help='Local server WebSocket URL')
    parser.add_argument('--bot-name', type=str, default='TrainingBot',
                        help='Bot username')
    parser.add_argument('--opponent-name', type=str, default='OpponentBot',
                        help='Opponent bot username')

    args = parser.parse_args()

    # Normalize format
    format_type = None
    if args.format:
        if args.format.lower() in ['s', 'singles']:
            format_type = 'S'
        elif args.format.lower() in ['d', 'doubles']:
            format_type = 'D'

    # Load teams
    print(f"Loading teams from {args.teams_dir}...")
    team_loader = TieredTeamLoader(args.teams_dir)

    if team_loader.get_team_count() == 0:
        print("ERROR: No teams loaded! Please check your teams directory.")
        return

    print(f"[OK] Loaded {team_loader.get_team_count()} teams")

    # Check if requested tier/format has teams
    if args.tier or format_type:
        count = team_loader.get_tier_count(args.tier, format_type)
        if count < 2:
            print(f"ERROR: Not enough teams for Tier {args.tier} {format_type}")
            print(f"Found only {count} teams, need at least 2")
            return

    # Create agent
    print("Initializing agent...")
    agent = BattleAgent(device=args.device)

    # Load checkpoint if specified
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        agent.load(args.checkpoint)

    # Create trainer
    trainer = TieredBattleTrainer(
        agent=agent,
        team_loader=team_loader,
        bot_username=args.bot_name,
        opponent_username=args.opponent_name,
        server_url=args.server,
        tier=args.tier,
        format_type=format_type
    )

    # Train
    try:
        asyncio.run(trainer.train(num_battles=args.battles))
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print(f"Battles completed: {trainer.battle_count}")
        print(f"Win rate: {trainer.wins}/{trainer.battle_count}")

        # Save final checkpoint
        trainer.save_checkpoint()
        trainer.save_stats()
        trainer.print_tier_stats()
        print("Final checkpoint saved")


if __name__ == "__main__":
    main()
