"""
Tier-Aware Team Loader for BDSP Battle Tower Teams
Loads teams with tier (set_number) and format (Singles/Doubles) metadata
"""

import os
import csv
import random
from typing import List, Dict, Any, Optional
import re


class TieredTeamLoader:
    """Loads and manages Pokemon teams with tier and format awareness"""

    def __init__(self, teams_directory: str = "bdsp_BT_teams"):
        """
        Initialize tier-aware team loader

        Args:
            teams_directory: Directory containing team files and summary.csv
        """
        self.teams_directory = teams_directory
        self.teams = []
        self.team_files = []
        self.summary_data = {}  # Map filename -> {tier, format, ...}

        # Load summary metadata first
        self._load_summary()
        # Then load teams
        self._load_all_teams()

    def _load_summary(self):
        """Load tier and format information from summary.csv"""
        summary_path = os.path.join(self.teams_directory, 'summary.csv')

        if not os.path.exists(summary_path):
            print(f"Warning: summary.csv not found in {self.teams_directory}")
            print("Teams will be loaded without tier/format information")
            return

        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row['file_name']
                    self.summary_data[filename] = {
                        'tier': int(row['set_number']),  # 1-7
                        'format': row['format'],  # 'S' or 'D'
                        'trainer_name': row['trainer_name'],
                        'trainer_class': row['trainer_class'],
                        'master_or_normal': row['master_or_normal'],
                        'team_id': row['team_id'],
                        'team_label': row['team_label'],
                    }

            print(f"Loaded metadata for {len(self.summary_data)} teams from summary.csv")

        except Exception as e:
            print(f"Error loading summary.csv: {e}")

    def _load_all_teams(self):
        """Load all teams from the directory"""
        if not os.path.exists(self.teams_directory):
            print(f"Warning: Teams directory '{self.teams_directory}' not found")
            return

        # Get all team files
        for filename in os.listdir(self.teams_directory):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.teams_directory, filename)
                self.team_files.append(filepath)

        print(f"Found {len(self.team_files)} team files")

        # Load teams
        for filepath in self.team_files:
            try:
                filename = os.path.basename(filepath)
                team_data = self._load_team_file(filepath)

                if team_data:
                    # Get metadata from summary
                    metadata = self.summary_data.get(filename, {})

                    team_info = {
                        'filepath': filepath,
                        'filename': filename,
                        'team_text': team_data['text'],
                        'pokemon': team_data['pokemon'],
                        'tier': metadata.get('tier', 0),  # 0 if unknown
                        'format': metadata.get('format', 'D'),  # Default to Doubles
                        'trainer_name': metadata.get('trainer_name', 'Unknown'),
                        'trainer_class': metadata.get('trainer_class', 'Unknown'),
                        'master_or_normal': metadata.get('master_or_normal', 'N'),
                        'team_id': metadata.get('team_id', ''),
                        'team_label': metadata.get('team_label', ''),
                    }

                    self.teams.append(team_info)

            except Exception as e:
                print(f"Error loading {filepath}: {e}")

        print(f"Successfully loaded {len(self.teams)} teams")

        # Print tier distribution
        self._print_tier_distribution()

    def _print_tier_distribution(self):
        """Print distribution of teams across tiers and formats"""
        tier_counts = {}
        format_counts = {'S': 0, 'D': 0}

        for team in self.teams:
            tier = team['tier']
            format_type = team['format']

            key = f"Tier {tier} - {'Singles' if format_type == 'S' else 'Doubles'}"
            tier_counts[key] = tier_counts.get(key, 0) + 1
            format_counts[format_type] = format_counts.get(format_type, 0) + 1

        print("\nTeam Distribution:")
        print(f"  Singles: {format_counts['S']} teams")
        print(f"  Doubles: {format_counts['D']} teams")
        print("\nBy Tier:")
        for key in sorted(tier_counts.keys()):
            print(f"  {key}: {tier_counts[key]} teams")

    def _load_team_file(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load a single team file

        Args:
            filepath: Path to team file

        Returns:
            Dictionary with team data
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        if not content.strip():
            return None

        # Parse Pokemon from the team
        pokemon_list = []
        current_pokemon = []

        for line in content.split('\n'):
            line = line.strip()

            # Check if this is a new Pokemon (starts with species @ item)
            if '@' in line and not line.startswith('-'):
                # Save previous Pokemon if exists
                if current_pokemon:
                    pokemon_list.append('\n'.join(current_pokemon))

                # Start new Pokemon
                current_pokemon = [line]
            elif line:
                # Add to current Pokemon
                current_pokemon.append(line)

        # Add last Pokemon
        if current_pokemon:
            pokemon_list.append('\n'.join(current_pokemon))

        return {
            'text': content,
            'pokemon': pokemon_list
        }

    def get_random_team(self, tier: Optional[int] = None,
                       format_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get a random team, optionally filtered by tier and format

        Args:
            tier: Tier number (1-7), or None for any tier
            format_type: 'S' for Singles, 'D' for Doubles, or None for any format

        Returns:
            Team dictionary or None if no matching teams
        """
        filtered_teams = self._filter_teams(tier=tier, format_type=format_type)

        if not filtered_teams:
            return None

        return random.choice(filtered_teams)

    def get_matching_team_pair(self, tier: Optional[int] = None,
                               format_type: Optional[str] = None) -> Optional[tuple]:
        """
        Get two teams from the same tier and format for fair battles

        Args:
            tier: Tier number (1-7), or None for random tier
            format_type: 'S' for Singles, 'D' for Doubles, or None for random format

        Returns:
            Tuple of (team1, team2) or None if not enough matching teams
        """
        # If tier/format not specified, pick random from available
        if tier is None or format_type is None:
            # Get a random team to determine tier/format
            sample_team = self.get_random_team()
            if not sample_team:
                return None

            if tier is None:
                tier = sample_team['tier']
            if format_type is None:
                format_type = sample_team['format']

        # Get all teams matching tier and format
        filtered_teams = self._filter_teams(tier=tier, format_type=format_type)

        if len(filtered_teams) < 2:
            print(f"Warning: Not enough teams for Tier {tier} {format_type}")
            return None

        # Select two different teams
        team1, team2 = random.sample(filtered_teams, 2)

        return (team1, team2)

    def _filter_teams(self, tier: Optional[int] = None,
                     format_type: Optional[str] = None,
                     master_only: bool = False) -> List[Dict[str, Any]]:
        """
        Filter teams by criteria

        Args:
            tier: Tier number (1-7)
            format_type: 'S' for Singles, 'D' for Doubles
            master_only: If True, only return Master Class teams

        Returns:
            List of matching teams
        """
        filtered = self.teams

        if tier is not None:
            filtered = [t for t in filtered if t['tier'] == tier]

        if format_type is not None:
            filtered = [t for t in filtered if t['format'] == format_type]

        if master_only:
            filtered = [t for t in filtered if t['master_or_normal'] == 'M']

        return filtered

    def get_team_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """Get team by index"""
        if 0 <= index < len(self.teams):
            return self.teams[index]
        return None

    def get_team_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get team by filename"""
        for team in self.teams:
            if team['filename'] == filename:
                return team
        return None

    def get_all_teams(self) -> List[Dict[str, Any]]:
        """Get all loaded teams"""
        return self.teams

    def get_team_count(self) -> int:
        """Get number of loaded teams"""
        return len(self.teams)

    def get_tier_count(self, tier: int, format_type: str = None) -> int:
        """Get number of teams in a specific tier"""
        filtered = self._filter_teams(tier=tier, format_type=format_type)
        return len(filtered)

    def export_team_for_showdown(self, team: Dict[str, Any]) -> str:
        """Export team in Showdown format"""
        return team['team_text']

    def parse_pokemon_species(self, pokemon_text: str) -> str:
        """Extract species name from Pokemon text"""
        first_line = pokemon_text.split('\n')[0]
        species = first_line.split('@')[0].strip()
        return species

    def parse_pokemon_item(self, pokemon_text: str) -> str:
        """Extract item from Pokemon text"""
        first_line = pokemon_text.split('\n')[0]
        if '@' in first_line:
            item = first_line.split('@')[1].strip()
            return item
        return ""

    def get_tier_stats(self) -> Dict[int, Dict[str, int]]:
        """
        Get statistics for each tier

        Returns:
            Dictionary mapping tier -> {singles: count, doubles: count}
        """
        stats = {}

        for tier in range(1, 8):  # Tiers 1-7
            stats[tier] = {
                'singles': len(self._filter_teams(tier=tier, format_type='S')),
                'doubles': len(self._filter_teams(tier=tier, format_type='D')),
            }

        return stats


def test_tiered_loader():
    """Test the tiered team loader"""
    loader = TieredTeamLoader()

    print(f"\n{'='*60}")
    print("Tier Statistics")
    print('='*60)

    tier_stats = loader.get_tier_stats()
    for tier, counts in tier_stats.items():
        total = counts['singles'] + counts['doubles']
        print(f"Tier {tier}: {total:4d} teams ({counts['singles']:4d} Singles, {counts['doubles']:4d} Doubles)")

    # Test getting matching pairs
    print(f"\n{'='*60}")
    print("Testing Matching Team Pairs")
    print('='*60)

    for tier in [1, 3, 5, 7]:
        for format_type in ['S', 'D']:
            pair = loader.get_matching_team_pair(tier=tier, format_type=format_type)
            if pair:
                team1, team2 = pair
                format_name = 'Singles' if format_type == 'S' else 'Doubles'
                print(f"\nTier {tier} {format_name}:")
                print(f"  Team 1: {team1['filename']}")
                print(f"  Team 2: {team2['filename']}")

    # Test random matching pair (any tier, any format but must match)
    print(f"\n{'='*60}")
    print("Random Matched Pair")
    print('='*60)

    pair = loader.get_matching_team_pair()
    if pair:
        team1, team2 = pair
        format_name = 'Singles' if team1['format'] == 'S' else 'Doubles'
        print(f"Tier {team1['tier']} {format_name}:")
        print(f"  Team 1: {team1['trainer_name']} - {team1['filename']}")
        print(f"  Team 2: {team2['trainer_name']} - {team2['filename']}")


if __name__ == "__main__":
    test_tiered_loader()
