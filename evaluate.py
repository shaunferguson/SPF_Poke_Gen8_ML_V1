"""
Evaluation Script for Battle Agent
Test agent performance and compare team fitness
"""

import argparse
import json
import os
from typing import Dict, List, Tuple
import numpy as np

from battle_agent import BattleAgent
from team_loader import TeamLoader


class TeamEvaluator:
    """Evaluate and compare team fitness using trained agent"""

    def __init__(self, agent: BattleAgent, team_loader: TeamLoader):
        """
        Initialize evaluator

        Args:
            agent: Trained battle agent
            team_loader: Team loader
        """
        self.agent = agent
        self.team_loader = team_loader
        self.results = {}

    def evaluate_team(self, team: Dict, num_battles: int = 50) -> Dict:
        """
        Evaluate a single team's performance

        Args:
            team: Team dictionary
            num_battles: Number of battles to simulate

        Returns:
            Dictionary with team evaluation results
        """
        print(f"Evaluating team: {team['filename']}")

        wins = 0
        losses = 0
        total_turns = 0
        total_damage_dealt = 0
        total_damage_taken = 0

        # This requires battle simulation implementation
        # For now, return placeholder results
        print(f"  Battle simulation not yet implemented")
        print(f"  Planned: {num_battles} battles")

        return {
            'team_name': team['filename'],
            'wins': wins,
            'losses': losses,
            'win_rate': wins / num_battles if num_battles > 0 else 0,
            'avg_turns': total_turns / num_battles if num_battles > 0 else 0,
            'avg_damage_dealt': total_damage_dealt / num_battles if num_battles > 0 else 0,
            'avg_damage_taken': total_damage_taken / num_battles if num_battles > 0 else 0,
        }

    def evaluate_all_teams(self, num_battles_per_team: int = 50) -> Dict[str, Dict]:
        """
        Evaluate all teams

        Args:
            num_battles_per_team: Number of battles for each team

        Returns:
            Dictionary mapping team names to evaluation results
        """
        all_teams = self.team_loader.get_all_teams()
        print(f"Evaluating {len(all_teams)} teams with {num_battles_per_team} battles each")

        results = {}
        for i, team in enumerate(all_teams):
            print(f"\nProgress: {i+1}/{len(all_teams)}")
            results[team['filename']] = self.evaluate_team(team, num_battles_per_team)

        self.results = results
        return results

    def get_team_rankings(self, metric: str = 'win_rate') -> List[Tuple[str, float]]:
        """
        Get team rankings by specified metric

        Args:
            metric: Metric to rank by ('win_rate', 'avg_turns', etc.)

        Returns:
            List of (team_name, metric_value) tuples sorted by metric
        """
        if not self.results:
            print("No results available. Run evaluate_all_teams first.")
            return []

        rankings = [
            (team_name, data[metric])
            for team_name, data in self.results.items()
        ]

        # Sort descending for win_rate, ascending for others
        reverse = metric == 'win_rate'
        rankings.sort(key=lambda x: x[1], reverse=reverse)

        return rankings

    def print_rankings(self, top_n: int = 10):
        """
        Print top teams by win rate

        Args:
            top_n: Number of top teams to display
        """
        rankings = self.get_team_rankings('win_rate')

        print(f"\n{'='*60}")
        print(f"Top {top_n} Teams by Win Rate")
        print(f"{'='*60}")

        for i, (team_name, win_rate) in enumerate(rankings[:top_n], 1):
            result = self.results[team_name]
            print(f"{i:2d}. {team_name}")
            print(f"    Win Rate: {win_rate*100:.1f}% ({result['wins']}/{result['wins']+result['losses']})")
            print(f"    Avg Turns: {result['avg_turns']:.1f}")
            print(f"    Avg Damage Dealt: {result['avg_damage_dealt']:.1f}")
            print(f"    Avg Damage Taken: {result['avg_damage_taken']:.1f}")
            print()

    def save_results(self, filepath: str):
        """
        Save evaluation results to file

        Args:
            filepath: Path to save results
        """
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filepath}")

    def load_results(self, filepath: str):
        """
        Load evaluation results from file

        Args:
            filepath: Path to load results from
        """
        with open(filepath, 'r') as f:
            self.results = json.load(f)
        print(f"Results loaded from {filepath}")


class AgentComparison:
    """Compare multiple trained agents"""

    def __init__(self, team_loader: TeamLoader):
        """
        Initialize agent comparison

        Args:
            team_loader: Team loader
        """
        self.team_loader = team_loader
        self.agents = {}

    def add_agent(self, name: str, checkpoint_path: str):
        """
        Add agent to comparison

        Args:
            name: Agent name/identifier
            checkpoint_path: Path to agent checkpoint
        """
        agent = BattleAgent()
        agent.load(checkpoint_path)
        self.agents[name] = agent
        print(f"Loaded agent '{name}' from {checkpoint_path}")

    def head_to_head(self, agent1_name: str, agent2_name: str,
                     num_battles: int = 100) -> Dict:
        """
        Run head-to-head battles between two agents

        Args:
            agent1_name: Name of first agent
            agent2_name: Name of second agent
            num_battles: Number of battles

        Returns:
            Dictionary with battle results
        """
        if agent1_name not in self.agents or agent2_name not in self.agents:
            print("One or both agents not found!")
            return {}

        print(f"\nHead-to-Head: {agent1_name} vs {agent2_name}")
        print(f"Battles: {num_battles}")

        # This requires battle simulation
        # Placeholder results
        agent1_wins = 0
        agent2_wins = 0

        print("Battle simulation not yet implemented")

        return {
            'agent1': agent1_name,
            'agent2': agent2_name,
            'agent1_wins': agent1_wins,
            'agent2_wins': agent2_wins,
            'agent1_win_rate': agent1_wins / num_battles if num_battles > 0 else 0,
            'agent2_win_rate': agent2_wins / num_battles if num_battles > 0 else 0,
        }

    def round_robin(self, num_battles_per_matchup: int = 50) -> Dict:
        """
        Run round-robin tournament between all agents

        Args:
            num_battles_per_matchup: Number of battles for each matchup

        Returns:
            Dictionary with tournament results
        """
        agent_names = list(self.agents.keys())

        if len(agent_names) < 2:
            print("Need at least 2 agents for round-robin!")
            return {}

        print(f"\nRound-Robin Tournament")
        print(f"Agents: {', '.join(agent_names)}")
        print(f"Battles per matchup: {num_battles_per_matchup}")

        results = {}
        for i, agent1 in enumerate(agent_names):
            for agent2 in agent_names[i+1:]:
                matchup_result = self.head_to_head(
                    agent1, agent2, num_battles_per_matchup
                )
                results[f"{agent1}_vs_{agent2}"] = matchup_result

        return results


def analyze_team_composition():
    """Analyze team composition trends from loaded teams"""
    loader = TeamLoader()

    print("\n" + "="*60)
    print("Team Composition Analysis")
    print("="*60)

    # Pokemon usage
    print("\nTop 20 Most Used Pokemon:")
    usage = loader.get_pokemon_usage_stats()
    for i, (species, count) in enumerate(list(usage.items())[:20], 1):
        percentage = (count / loader.get_team_count()) * 100
        print(f"{i:2d}. {species:20s} - {count:3d} times ({percentage:.1f}%)")

    # Item usage
    print("\nTop 15 Most Used Items:")
    item_usage = loader.get_item_usage_stats()
    for i, (item, count) in enumerate(list(item_usage.items())[:15], 1):
        percentage = (count / loader.get_team_count()) * 100
        print(f"{i:2d}. {item:20s} - {count:3d} times ({percentage:.1f}%)")

    # Type distribution (would require full implementation)
    print("\nType distribution analysis requires full Pokemon data implementation")


def main():
    """Main evaluation entry point"""
    parser = argparse.ArgumentParser(description="Evaluate Pokemon battle agent")
    parser.add_argument('--mode', type=str, default='analyze',
                        choices=['evaluate', 'compare', 'analyze'],
                        help='Evaluation mode')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to agent checkpoint')
    parser.add_argument('--teams-dir', type=str, default='bdsp_BT_teams',
                        help='Directory containing team files')
    parser.add_argument('--battles', type=int, default=50,
                        help='Number of battles per team/matchup')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Output file for results')
    parser.add_argument('--checkpoints', type=str, nargs='+',
                        help='Multiple checkpoints for comparison mode')

    args = parser.parse_args()

    # Load teams
    print(f"Loading teams from {args.teams_dir}...")
    team_loader = TeamLoader(args.teams_dir)

    if team_loader.get_team_count() == 0:
        print("No teams loaded!")
        return

    if args.mode == 'analyze':
        # Analyze team composition
        analyze_team_composition()

    elif args.mode == 'evaluate':
        # Evaluate single agent
        if not args.checkpoint:
            print("Please specify --checkpoint for evaluation mode")
            return

        print(f"\nLoading agent from {args.checkpoint}")
        agent = BattleAgent()
        agent.load(args.checkpoint)

        # Show agent stats
        print(f"\nAgent Stats: {agent.get_stats()}")

        # Evaluate teams
        evaluator = TeamEvaluator(agent, team_loader)
        results = evaluator.evaluate_all_teams(num_battles_per_team=args.battles)

        # Print rankings
        evaluator.print_rankings(top_n=20)

        # Save results
        evaluator.save_results(args.output)

    elif args.mode == 'compare':
        # Compare multiple agents
        if not args.checkpoints or len(args.checkpoints) < 2:
            print("Please specify at least 2 checkpoints with --checkpoints for compare mode")
            return

        comparison = AgentComparison(team_loader)

        # Load agents
        for i, checkpoint in enumerate(args.checkpoints, 1):
            name = f"Agent_{i}"
            comparison.add_agent(name, checkpoint)

        # Run round-robin
        results = comparison.round_robin(num_battles_per_matchup=args.battles)

        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nComparison results saved to {args.output}")


if __name__ == "__main__":
    main()
