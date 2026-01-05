"""
Team Loader for BDSP Battle Tower Teams
Loads and manages Pokemon teams from text files
"""

import os
import random
from typing import List, Dict, Any, Optional
import re


class TeamLoader:
    """Loads and manages Pokemon teams from BDSP Battle Tower format"""

    def __init__(self, teams_directory: str = "bdsp_BT_teams"):
        """
        Initialize team loader

        Args:
            teams_directory: Directory containing team files
        """
        self.teams_directory = teams_directory
        self.teams = []
        self.team_files = []
        self._load_all_teams()

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
                team_data = self._load_team_file(filepath)
                if team_data:
                    self.teams.append({
                        'filepath': filepath,
                        'filename': os.path.basename(filepath),
                        'team_text': team_data['text'],
                        'pokemon': team_data['pokemon']
                    })
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

        print(f"Successfully loaded {len(self.teams)} teams")

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

    def get_random_team(self) -> Optional[Dict[str, Any]]:
        """
        Get a random team

        Returns:
            Team dictionary or None if no teams loaded
        """
        if not self.teams:
            return None

        return random.choice(self.teams)

    def get_team_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Get team by index

        Args:
            index: Team index

        Returns:
            Team dictionary or None if index out of range
        """
        if 0 <= index < len(self.teams):
            return self.teams[index]
        return None

    def get_team_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get team by filename

        Args:
            filename: Filename to search for

        Returns:
            Team dictionary or None if not found
        """
        for team in self.teams:
            if team['filename'] == filename:
                return team
        return None

    def get_all_teams(self) -> List[Dict[str, Any]]:
        """
        Get all loaded teams

        Returns:
            List of team dictionaries
        """
        return self.teams

    def parse_pokemon_species(self, pokemon_text: str) -> str:
        """
        Extract species name from Pokemon text

        Args:
            pokemon_text: Pokemon text block

        Returns:
            Species name
        """
        first_line = pokemon_text.split('\n')[0]
        # Format: "Species @ Item"
        species = first_line.split('@')[0].strip()
        return species

    def parse_pokemon_item(self, pokemon_text: str) -> str:
        """
        Extract item from Pokemon text

        Args:
            pokemon_text: Pokemon text block

        Returns:
            Item name
        """
        first_line = pokemon_text.split('\n')[0]
        # Format: "Species @ Item"
        if '@' in first_line:
            item = first_line.split('@')[1].strip()
            return item
        return ""

    def parse_pokemon_ability(self, pokemon_text: str) -> str:
        """
        Extract ability from Pokemon text

        Args:
            pokemon_text: Pokemon text block

        Returns:
            Ability name
        """
        for line in pokemon_text.split('\n'):
            if line.startswith('Ability:'):
                return line.split('Ability:')[1].strip()
        return ""

    def parse_pokemon_moves(self, pokemon_text: str) -> List[str]:
        """
        Extract moves from Pokemon text

        Args:
            pokemon_text: Pokemon text block

        Returns:
            List of move names
        """
        moves = []
        for line in pokemon_text.split('\n'):
            if line.startswith('-'):
                move = line[1:].strip()
                moves.append(move)
        return moves

    def parse_team_info(self, team: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse detailed information from team

        Args:
            team: Team dictionary

        Returns:
            Dictionary with parsed team information
        """
        pokemon_info = []

        for pokemon_text in team['pokemon']:
            info = {
                'species': self.parse_pokemon_species(pokemon_text),
                'item': self.parse_pokemon_item(pokemon_text),
                'ability': self.parse_pokemon_ability(pokemon_text),
                'moves': self.parse_pokemon_moves(pokemon_text),
                'text': pokemon_text
            }
            pokemon_info.append(info)

        return {
            'filename': team['filename'],
            'filepath': team['filepath'],
            'team_text': team['team_text'],
            'pokemon': pokemon_info
        }

    def create_random_team_subset(self, num_pokemon: int = 4) -> Optional[str]:
        """
        Create a random team with a subset of Pokemon from loaded teams

        Args:
            num_pokemon: Number of Pokemon to include (default: 4 for doubles)

        Returns:
            Team text or None
        """
        if not self.teams:
            return None

        # Collect all individual Pokemon from all teams
        all_pokemon = []
        for team in self.teams:
            all_pokemon.extend(team['pokemon'])

        # Select random Pokemon
        selected = random.sample(all_pokemon, min(num_pokemon, len(all_pokemon)))

        # Combine into team text
        return '\n\n'.join(selected)

    def export_team_for_showdown(self, team: Dict[str, Any]) -> str:
        """
        Export team in Showdown format

        Args:
            team: Team dictionary

        Returns:
            Team text formatted for Showdown
        """
        return team['team_text']

    def get_team_count(self) -> int:
        """Get number of loaded teams"""
        return len(self.teams)

    def filter_teams_by_pokemon(self, pokemon_name: str) -> List[Dict[str, Any]]:
        """
        Filter teams that contain a specific Pokemon

        Args:
            pokemon_name: Pokemon species name to search for

        Returns:
            List of teams containing that Pokemon
        """
        matching_teams = []

        for team in self.teams:
            team_info = self.parse_team_info(team)
            for pokemon in team_info['pokemon']:
                if pokemon_name.lower() in pokemon['species'].lower():
                    matching_teams.append(team)
                    break

        return matching_teams

    def get_pokemon_usage_stats(self) -> Dict[str, int]:
        """
        Get usage statistics for all Pokemon across teams

        Returns:
            Dictionary mapping Pokemon species to usage count
        """
        usage = {}

        for team in self.teams:
            for pokemon_text in team['pokemon']:
                species = self.parse_pokemon_species(pokemon_text)
                usage[species] = usage.get(species, 0) + 1

        return dict(sorted(usage.items(), key=lambda x: x[1], reverse=True))

    def get_item_usage_stats(self) -> Dict[str, int]:
        """
        Get usage statistics for items across teams

        Returns:
            Dictionary mapping items to usage count
        """
        usage = {}

        for team in self.teams:
            for pokemon_text in team['pokemon']:
                item = self.parse_pokemon_item(pokemon_text)
                if item:
                    usage[item] = usage.get(item, 0) + 1

        return dict(sorted(usage.items(), key=lambda x: x[1], reverse=True))


def test_team_loader():
    """Test the team loader"""
    loader = TeamLoader()

    print(f"\nTotal teams loaded: {loader.get_team_count()}")

    # Get random team
    random_team = loader.get_random_team()
    if random_team:
        print(f"\nRandom team: {random_team['filename']}")
        team_info = loader.parse_team_info(random_team)

        print("\nPokemon in team:")
        for pokemon in team_info['pokemon']:
            print(f"  - {pokemon['species']} @ {pokemon['item']}")
            print(f"    Ability: {pokemon['ability']}")
            print(f"    Moves: {', '.join(pokemon['moves'])}")

    # Get usage stats
    print("\nTop 10 most used Pokemon:")
    usage = loader.get_pokemon_usage_stats()
    for i, (species, count) in enumerate(list(usage.items())[:10]):
        print(f"  {i+1}. {species}: {count} times")

    print("\nTop 10 most used items:")
    item_usage = loader.get_item_usage_stats()
    for i, (item, count) in enumerate(list(item_usage.items())[:10]):
        print(f"  {i+1}. {item}: {count} times")

    # Test filtering
    garchomp_teams = loader.filter_teams_by_pokemon('Garchomp')
    print(f"\nTeams with Garchomp: {len(garchomp_teams)}")


if __name__ == "__main__":
    test_team_loader()
