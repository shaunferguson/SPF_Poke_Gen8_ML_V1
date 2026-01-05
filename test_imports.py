"""
Test script to verify all components can be imported and initialized
Run this to ensure everything is set up correctly
"""

import sys
import traceback

def test_component(name, import_func):
    """Test importing and initializing a component"""
    try:
        print(f"Testing {name}...", end=" ")
        import_func()
        print("[OK]")
        return True
    except Exception as e:
        print("[FAILED]")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False

def test_all():
    """Test all components"""
    results = []

    print("\n" + "="*60)
    print("Testing Pokemon Battle AI Components")
    print("="*60 + "\n")

    # Test imports
    print("1. Testing Imports:")
    print("-" * 60)

    def test_torch():
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")

    results.append(test_component("PyTorch", test_torch))

    def test_websockets():
        import websockets
        print(f"  WebSockets version: {websockets.__version__}")

    results.append(test_component("WebSockets", test_websockets))

    def test_numpy():
        import numpy as np
        print(f"  NumPy version: {np.__version__}")

    results.append(test_component("NumPy", test_numpy))

    # Test modules
    print("\n2. Testing Custom Modules:")
    print("-" * 60)

    def test_team_loader():
        from team_loader import TeamLoader
        loader = TeamLoader()
        print(f"  Loaded {loader.get_team_count()} teams")

    results.append(test_component("Team Loader", test_team_loader))

    def test_state_encoder():
        from state_encoder import BattleStateEncoder
        encoder = BattleStateEncoder()
        print(f"  State size: {encoder.state_size}")

    results.append(test_component("State Encoder", test_state_encoder))

    def test_dqn_model():
        from dqn_model import DQN, DuelingDQN, ReplayBuffer
        import torch
        model = DuelingDQN(state_size=200, action_size=9)
        sample_input = torch.randn(1, 200)
        output = model(sample_input)
        print(f"  Model output shape: {output.shape}")

    results.append(test_component("DQN Model", test_dqn_model))

    def test_battle_agent():
        from battle_agent import BattleAgent
        agent = BattleAgent()
        stats = agent.get_stats()
        print(f"  Agent initialized with epsilon: {stats['epsilon']:.2f}")

    results.append(test_component("Battle Agent", test_battle_agent))

    def test_showdown_client():
        from showdown_client import ShowdownClient
        client = ShowdownClient(username="TestBot")
        print(f"  Client created for user: {client.username}")

    results.append(test_component("Showdown Client", test_showdown_client))

    # Test integration
    print("\n3. Testing Integration:")
    print("-" * 60)

    def test_full_pipeline():
        from team_loader import TeamLoader
        from state_encoder import BattleStateEncoder
        from battle_agent import BattleAgent

        # Load team
        loader = TeamLoader()
        if loader.get_team_count() == 0:
            print("  Warning: No teams loaded")
            return

        team = loader.get_random_team()
        print(f"  Loaded team: {team['filename']}")

        # Create encoder
        encoder = BattleStateEncoder()

        # Create agent
        agent = BattleAgent()

        # Test with dummy state
        dummy_state = {
            'team': {
                'active': {'species': 'Garchomp', 'condition': '100/100'},
                'pokemon': [
                    {'species': 'Garchomp', 'condition': '100/100'},
                ]
            },
            'opponent': {
                'active': {'species': 'Charizard', 'condition': '100/100'},
                'pokemon': [
                    {'species': 'Charizard', 'condition': '100/100'},
                ]
            },
            'field': {},
            'request': {
                'active': [{
                    'moves': [
                        {'move': 'Earthquake', 'pp': 10, 'maxpp': 10},
                    ]
                }],
                'side': {
                    'pokemon': [
                        {'active': True, 'condition': '100/100'},
                    ]
                }
            }
        }

        # Encode state
        state_tensor = encoder.encode_state(dummy_state)
        print(f"  Encoded state shape: {state_tensor.shape}")

        # Get action
        action = agent.select_action(dummy_state)
        print(f"  Selected action: {action}")

        print("  Full pipeline working!")

    results.append(test_component("Full Pipeline", test_full_pipeline))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(results)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n[SUCCESS] All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Install poke-env: pip install poke-env")
        print("2. Run: python poke_env_integration.py --mode train --battles 10")
        print("3. Check QUICKSTART.md for more information")
    else:
        print("\n[ERROR] Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("- Make sure you've run: pip install -r requirements.txt")
        print("- Check that bdsp_BT_teams directory exists and has team files")
        print("- Verify Python version is 3.8 or higher")

    print()
    return passed == total

if __name__ == "__main__":
    success = test_all()
    sys.exit(0 if success else 1)
