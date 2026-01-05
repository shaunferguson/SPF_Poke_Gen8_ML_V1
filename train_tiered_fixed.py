"""
Clean Tier-Aware Local Training with Ladder Queueing
"""

import asyncio
import argparse
import os
import random
from datetime import datetime

from showdown_client_fixed import ShowdownClient, pack_team
from battle_agent import BattleAgent
from team_loader_tiered import TieredTeamLoader


class TieredBattleTrainer:
    def __init__(self, agent, team_loader, tier=None, format_type=None):
        self.agent = agent
        self.team_loader = team_loader
        self.tier = tier
        self.format_type = format_type

        self.bot_client = ShowdownClient()
        self.opp_client = ShowdownClient()

        self.battle_count = 0
        self.wins = 0
        self.battle_ended = asyncio.Event()

    async def bot_callback(self, battle_id, state):
        # Plug in your full agent logic here later
        # For now: random moves so battles actually happen
        if 'request' in state['raw']:
            choices = ["move 1", "move 2", "move 3", "move 4", "switch 2", "switch 3"]
            choice = random.choice([c for c in choices if c in state['raw']])  # crude legal check
            await self.bot_client.make_move(battle_id, choice)

        if '|win|' in state['raw']:
            winner = state['raw'].split('|win|')[1].strip()
            if self.bot_client.username in winner:
                self.wins += 1
            self.battle_count += 1
            print(f"\nBattle {self.battle_count} done! Wins: {self.wins}/{self.battle_count}")
            self.battle_ended.set()

    async def opp_callback(self, battle_id, state):
        if 'request' in state['raw']:
            choices = ["move 1", "move 2", "move 3", "move 4", "switch 2", "switch 3"]
            choice = random.choice(choices)
            await self.opp_client.make_move(battle_id, choice)

        if '|win|' in state['raw']:
            self.battle_ended.set()

    async def train(self, num_battles):
        # Connect & login sequentially
        for client in [self.bot_client, self.opp_client]:
            await client.connect()
            await client.login()

        # Background listeners
        asyncio.create_task(self.bot_client.listen(self.bot_callback))
        asyncio.create_task(self.opp_client.listen(self.opp_callback))

        # UPDATE THIS TO YOUR ACTUAL FORMAT ID(S)!
        format_id = "gen8bdspbattletower"  # or gen8bdspsingles / gen8bdspdoubles

        while self.battle_count < num_battles:
            self.battle_ended.clear()

            pair = self.team_loader.get_matching_team_pair(tier=self.tier, format_type=self.format_type)
            if not pair:
                print("No more teams!")
                break

            bot_team, opp_team = pair

            print(f"\nQueueing Battle {self.battle_count + 1} | Tier {bot_team['tier']} {'Singles' if bot_team['format']=='S' else 'Doubles'}")
            print(f"  Bot: {bot_team['filename']}")
            print(f"  Opp: {opp_team['filename']}")

            bot_packed = pack_team(bot_team['team_text'])
            opp_packed = pack_team(opp_team['team_text'])

            await asyncio.gather(
                self.bot_client.send_command(f"/utm {bot_packed}"),
                self.opp_client.send_command(f"/utm {opp_packed}")
            )

            await asyncio.sleep(0.5)

            await asyncio.gather(
                self.bot_client.send_command(f"/search {format_id}"),
                self.opp_client.send_command(f"/search {format_id}")
            )

            await self.battle_ended.wait()

        print("\nAll done! 🎉")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--battles', type=int, default=100)
    parser.add_argument('--teams-dir', type=str, default='bdsp_BT_teams')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--tier', type=int, default=None)
    parser.add_argument('--format', type=str, default=None)

    args = parser.parse_args()

    format_type = None
    if args.format:
        format_type = 'S' if args.format.lower().startswith('s') else 'D'

    loader = TieredTeamLoader(args.teams_dir)
    print(f"Loaded {loader.get_team_count()} teams")

    agent = BattleAgent(device=args.device)
    if args.checkpoint:
        agent.load(args.checkpoint)

    trainer = TieredBattleTrainer(agent, loader, args.tier, format_type)
    asyncio.run(trainer.train(args.battles))


if __name__ == "__main__":
    main()