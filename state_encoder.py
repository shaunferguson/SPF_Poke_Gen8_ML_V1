"""
State Encoder for Pokemon Battle States
Converts battle state into neural network input tensors
"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple
import json


class BattleStateEncoder:
    """Encodes Pokemon battle state into tensor format for neural network"""

    # Gen 4 Pokemon (National Dex 1-493)
    POKEMON_COUNT = 493
    MAX_MOVES = 4
    MAX_TEAM_SIZE = 6

    # Type chart (18 types in Gen 4)
    TYPES = [
        'Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice',
        'Fighting', 'Poison', 'Ground', 'Flying', 'Psychic', 'Bug',
        'Rock', 'Ghost', 'Dragon', 'Dark', 'Steel', 'Fairy'
    ]
    TYPE_COUNT = len(TYPES)

    # Common status conditions
    STATUS = ['', 'psn', 'tox', 'brn', 'par', 'slp', 'frz']
    STATUS_COUNT = len(STATUS)

    # Volatile statuses (in-battle conditions)
    VOLATILE_STATUS = [
        'confusion', 'flinch', 'partiallytrapped', 'leechseed',
        'substitute', 'curse', 'nightmare', 'attract', 'torment',
        'disable', 'embargo', 'healblock', 'taunt', 'encore'
    ]
    VOLATILE_COUNT = len(VOLATILE_STATUS)

    # Weather conditions
    WEATHER = ['', 'sunnyday', 'raindance', 'sandstorm', 'hail']
    WEATHER_COUNT = len(WEATHER)

    # Stat stages (-6 to +6 = 13 possible values)
    STAT_STAGES = 13

    def __init__(self):
        """Initialize the encoder with Pokemon data"""
        # In a full implementation, load Pokemon stats, types, moves from a data file
        # For now, we'll use simplified encoding
        self.pokemon_to_id = {}
        self.move_to_id = {}
        self.ability_to_id = {}
        self.item_to_id = {}

        # Initialize ID mappings (these would be loaded from data files)
        self._initialize_mappings()

        # Calculate state vector size
        self.state_size = self._calculate_state_size()

    def _initialize_mappings(self):
        """Initialize mappings for Pokemon, moves, abilities, items"""
        # Placeholder - in production, load from comprehensive data files
        # These would include all Gen 4 Pokemon, moves, abilities, and items
        pass

    def _calculate_state_size(self) -> int:
        """
        Calculate the total size of the state vector

        Returns:
            Total state vector dimension
        """
        # Per-Pokemon features (for active Pokemon on each side)
        per_pokemon_features = (
            1 +  # HP percentage
            self.TYPE_COUNT * 2 +  # Types (one-hot encoded, dual types)
            self.STATUS_COUNT +  # Status condition (one-hot)
            self.VOLATILE_COUNT +  # Volatile statuses (binary)
            6 * self.STAT_STAGES +  # Stat stages (HP, Atk, Def, SpA, SpD, Spe)
            1  # Fainted flag
        )

        # Active Pokemon (player + opponent)
        active_pokemon_size = per_pokemon_features * 2

        # Team information (remaining Pokemon)
        team_info_size = (
            self.MAX_TEAM_SIZE * 2 +  # HP of each team member (player + opponent)
            self.MAX_TEAM_SIZE * 2  # Fainted flags
        )

        # Field conditions
        field_size = (
            self.WEATHER_COUNT +  # Weather (one-hot)
            1 +  # Weather turns remaining
            4 +  # Terrain (none, electric, grassy, misty, psychic - simplified for Gen 4)
            1 +  # Terrain turns remaining
            8  # Hazards (Stealth Rock, Spikes x3, Toxic Spikes x2, Reflect, Light Screen)
        )

        # Move information (available moves for active Pokemon)
        move_info_size = self.MAX_MOVES * (
            1 +  # PP remaining (normalized)
            1 +  # Base power (normalized)
            self.TYPE_COUNT  # Move type (one-hot)
        )

        total_size = (
            active_pokemon_size +
            team_info_size +
            field_size +
            move_info_size
        )

        return total_size

    def encode_state(self, battle_state: Dict[str, Any]) -> torch.Tensor:
        """
        Encode battle state into a tensor

        Args:
            battle_state: Battle state dictionary from ShowdownClient

        Returns:
            Tensor representing the battle state
        """
        state_vector = []

        # Encode active Pokemon (player)
        player_active = self._encode_pokemon(
            battle_state.get('team', {}).get('active'),
            battle_state.get('request', {})
        )
        state_vector.extend(player_active)

        # Encode active Pokemon (opponent)
        opponent_active = self._encode_pokemon(
            battle_state.get('opponent', {}).get('active'),
            {}
        )
        state_vector.extend(opponent_active)

        # Encode team information
        team_info = self._encode_team_info(battle_state)
        state_vector.extend(team_info)

        # Encode field conditions
        field_info = self._encode_field(battle_state.get('field', {}))
        state_vector.extend(field_info)

        # Encode available moves
        move_info = self._encode_moves(battle_state.get('request', {}))
        state_vector.extend(move_info)

        return torch.FloatTensor(state_vector)

    def _encode_pokemon(self, pokemon: Dict[str, Any], request_data: Dict[str, Any]) -> List[float]:
        """Encode a single Pokemon's state"""
        if not pokemon:
            # Return zero vector if no Pokemon
            per_pokemon_size = (
                1 + self.TYPE_COUNT * 2 + self.STATUS_COUNT +
                self.VOLATILE_COUNT + 6 * self.STAT_STAGES + 1
            )
            return [0.0] * per_pokemon_size

        vector = []

        # HP percentage
        hp_percent = self._parse_hp(pokemon.get('condition', '100/100'))
        vector.append(hp_percent)

        # Types (one-hot, allowing dual types)
        types = self._get_pokemon_types(pokemon.get('species', ''))
        type_vector = [0.0] * (self.TYPE_COUNT * 2)
        for i, ptype in enumerate(types[:2]):
            if ptype in self.TYPES:
                type_vector[i * self.TYPE_COUNT + self.TYPES.index(ptype)] = 1.0
        vector.extend(type_vector)

        # Status condition
        status = self._parse_status(pokemon.get('condition', ''))
        status_vector = [0.0] * self.STATUS_COUNT
        if status in self.STATUS:
            status_vector[self.STATUS.index(status)] = 1.0
        vector.extend(status_vector)

        # Volatile statuses
        volatile_vector = [0.0] * self.VOLATILE_COUNT
        volatiles = pokemon.get('volatiles', [])
        for i, vol_status in enumerate(self.VOLATILE_STATUS):
            if vol_status in volatiles:
                volatile_vector[i] = 1.0
        vector.extend(volatile_vector)

        # Stat stages (default to 0, which is +0 stage = index 6)
        stat_stages = pokemon.get('stat_stages', {})
        for stat in ['hp', 'atk', 'def', 'spa', 'spd', 'spe']:
            stage = stat_stages.get(stat, 0) + 6  # Convert -6..+6 to 0..12
            stage_vector = [0.0] * self.STAT_STAGES
            stage_vector[min(max(stage, 0), 12)] = 1.0
            vector.extend(stage_vector)

        # Fainted flag
        vector.append(1.0 if hp_percent == 0 else 0.0)

        return vector

    def _encode_team_info(self, battle_state: Dict[str, Any]) -> List[float]:
        """Encode team HP and faint status"""
        vector = []

        # Player team
        player_team = battle_state.get('team', {}).get('pokemon', [])
        for i in range(self.MAX_TEAM_SIZE):
            if i < len(player_team):
                hp = self._parse_hp(player_team[i].get('condition', '100/100'))
                vector.append(hp)
            else:
                vector.append(0.0)

        # Opponent team
        opponent_team = battle_state.get('opponent', {}).get('pokemon', [])
        for i in range(self.MAX_TEAM_SIZE):
            if i < len(opponent_team):
                hp = self._parse_hp(opponent_team[i].get('condition', '100/100'))
                vector.append(hp)
            else:
                vector.append(0.0)

        # Fainted flags (player)
        for i in range(self.MAX_TEAM_SIZE):
            if i < len(player_team):
                hp = self._parse_hp(player_team[i].get('condition', '100/100'))
                vector.append(1.0 if hp == 0 else 0.0)
            else:
                vector.append(0.0)

        # Fainted flags (opponent)
        for i in range(self.MAX_TEAM_SIZE):
            if i < len(opponent_team):
                hp = self._parse_hp(opponent_team[i].get('condition', '100/100'))
                vector.append(1.0 if hp == 0 else 0.0)
            else:
                vector.append(0.0)

        return vector

    def _encode_field(self, field: Dict[str, Any]) -> List[float]:
        """Encode field conditions (weather, terrain, hazards)"""
        vector = []

        # Weather
        weather = field.get('weather', '')
        weather_vector = [0.0] * self.WEATHER_COUNT
        if weather in self.WEATHER:
            weather_vector[self.WEATHER.index(weather)] = 1.0
        else:
            weather_vector[0] = 1.0  # No weather
        vector.extend(weather_vector)

        # Weather turns
        vector.append(field.get('weather_turns', 0) / 8.0)  # Normalize to [0, 1]

        # Terrain (simplified for Gen 4 - mostly unused)
        terrain_vector = [0.0] * 4
        terrain_vector[0] = 1.0  # No terrain by default in Gen 4
        vector.extend(terrain_vector)

        # Terrain turns
        vector.append(0.0)

        # Hazards (player side)
        player_hazards = field.get('player_hazards', {})
        vector.append(1.0 if player_hazards.get('stealthrock', False) else 0.0)
        vector.append(player_hazards.get('spikes', 0) / 3.0)
        vector.append(player_hazards.get('toxicspikes', 0) / 2.0)
        vector.append(1.0 if player_hazards.get('reflect', False) else 0.0)
        vector.append(1.0 if player_hazards.get('lightscreen', False) else 0.0)

        # Hazards (opponent side)
        opponent_hazards = field.get('opponent_hazards', {})
        vector.append(1.0 if opponent_hazards.get('stealthrock', False) else 0.0)
        vector.append(opponent_hazards.get('spikes', 0) / 3.0)
        vector.append(opponent_hazards.get('toxicspikes', 0) / 2.0)

        return vector

    def _encode_moves(self, request_data: Dict[str, Any]) -> List[float]:
        """Encode available moves"""
        vector = []

        active_data = request_data.get('active', [{}])[0] if request_data.get('active') else {}
        moves = active_data.get('moves', [])

        for i in range(self.MAX_MOVES):
            if i < len(moves):
                move = moves[i]

                # PP remaining (normalized)
                pp = move.get('pp', 0)
                max_pp = move.get('maxpp', 1)
                vector.append(pp / max_pp if max_pp > 0 else 0.0)

                # Base power (normalized to [0, 1], assuming max 250)
                base_power = self._get_move_base_power(move.get('move', ''))
                vector.append(base_power / 250.0)

                # Move type
                move_type = self._get_move_type(move.get('move', ''))
                type_vector = [0.0] * self.TYPE_COUNT
                if move_type in self.TYPES:
                    type_vector[self.TYPES.index(move_type)] = 1.0
                vector.extend(type_vector)
            else:
                # No move in this slot
                vector.append(0.0)  # PP
                vector.append(0.0)  # Base power
                vector.extend([0.0] * self.TYPE_COUNT)  # Type

        return vector

    def get_action_mask(self, battle_state: Dict[str, Any]) -> torch.Tensor:
        """
        Get mask of legal actions

        Args:
            battle_state: Battle state dictionary

        Returns:
            Binary tensor where 1 = legal action, 0 = illegal
        """
        # Maximum actions: 4 moves + 5 switches (can't switch to active) = 9
        max_actions = self.MAX_MOVES + (self.MAX_TEAM_SIZE - 1)
        mask = torch.zeros(max_actions, dtype=torch.float32)

        request_data = battle_state.get('request', {})

        if not request_data:
            return mask

        # Enable legal moves
        active_data = request_data.get('active', [{}])[0] if request_data.get('active') else {}
        moves = active_data.get('moves', [])

        for i, move in enumerate(moves):
            if not move.get('disabled', False) and move.get('pp', 0) > 0:
                mask[i] = 1.0

        # Enable legal switches
        side_data = request_data.get('side', {})
        pokemon_list = side_data.get('pokemon', [])

        for i, pokemon in enumerate(pokemon_list):
            if not pokemon.get('active', False) and pokemon.get('condition', '').split()[0] != '0':
                # Can switch to this Pokemon (not active and not fainted)
                # Switch actions start after move actions
                mask[self.MAX_MOVES + i] = 1.0

        return mask

    @staticmethod
    def _parse_hp(condition: str) -> float:
        """Parse HP from condition string (e.g., '50/100' -> 0.5)"""
        try:
            if not condition or condition == '0 fnt':
                return 0.0
            hp_part = condition.split()[0]
            if '/' in hp_part:
                current, max_hp = hp_part.split('/')
                return float(current) / float(max_hp)
            return 1.0
        except (ValueError, ZeroDivisionError):
            return 1.0

    @staticmethod
    def _parse_status(condition: str) -> str:
        """Parse status condition from condition string"""
        try:
            parts = condition.split()
            if len(parts) > 1:
                return parts[1]
            return ''
        except Exception:
            return ''

    def _get_pokemon_types(self, species: str) -> List[str]:
        """Get Pokemon types (placeholder - would use actual data)"""
        # In production, load from comprehensive Pokemon data
        # For now, return Normal type as default
        return ['Normal']

    def _get_move_base_power(self, move: str) -> float:
        """Get move base power (placeholder - would use actual data)"""
        # In production, load from move data
        return 80.0  # Default

    def _get_move_type(self, move: str) -> str:
        """Get move type (placeholder - would use actual data)"""
        # In production, load from move data
        return 'Normal'  # Default


def test_encoder():
    """Test the encoder with sample battle state"""
    encoder = BattleStateEncoder()

    sample_state = {
        'team': {
            'active': {
                'species': 'Garchomp',
                'condition': '75/100'
            },
            'pokemon': [
                {'species': 'Garchomp', 'condition': '75/100'},
                {'species': 'Ninetales', 'condition': '100/100'},
            ]
        },
        'opponent': {
            'active': {
                'species': 'Charizard',
                'condition': '50/100'
            },
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
            }]
        }
    }

    state_tensor = encoder.encode_state(sample_state)
    action_mask = encoder.get_action_mask(sample_state)

    print(f"State size: {encoder.state_size}")
    print(f"Encoded state shape: {state_tensor.shape}")
    print(f"Action mask shape: {action_mask.shape}")
    print(f"Legal actions: {action_mask.nonzero().squeeze()}")


if __name__ == "__main__":
    test_encoder()
