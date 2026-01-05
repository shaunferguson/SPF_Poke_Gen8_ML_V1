# Tier-Aware Training Guide

## Overview

Your Pokemon teams are organized into **7 tiers** and **2 formats** (Singles/Doubles). The training system now ensures that battles only occur between teams of the same tier and format for fair matchmaking!

## Team Distribution

**Total Teams:** 2019

### By Tier
- **Tier 1:** 805 teams (422 Singles, 383 Doubles)
- **Tier 2:** 200 teams (109 Singles, 91 Doubles)
- **Tier 3:** 201 teams (112 Singles, 89 Doubles)
- **Tier 4:** 200 teams (114 Singles, 86 Doubles)
- **Tier 5:** 200 teams (108 Singles, 92 Doubles)
- **Tier 6:** 208 teams (110 Singles, 98 Doubles)
- **Tier 7:** 205 teams (108 Singles, 97 Doubles)

### By Format
- **Singles:** 1083 teams (53.7%)
- **Doubles:** 936 teams (46.3%)

## Training Commands

### Train on All Tiers & Formats (Recommended)
```bash
python train_tiered.py --battles 1000
```

This will randomly select teams from all tiers and formats, but always match teams of the same tier/format against each other.

### Train on Specific Tier

**Tier 1 only:**
```bash
python train_tiered.py --battles 500 --tier 1
```

**Tier 7 only (hardest):**
```bash
python train_tiered.py --battles 500 --tier 7
```

### Train on Specific Format

**Singles only:**
```bash
python train_tiered.py --battles 1000 --format singles
```

**Doubles only:**
```bash
python train_tiered.py --battles 1000 --format doubles
```

### Train on Specific Tier AND Format

**Tier 1 Doubles:**
```bash
python train_tiered.py --battles 500 --tier 1 --format doubles
```

**Tier 5 Singles:**
```bash
python train_tiered.py --battles 500 --tier 5 --format singles
```

### Resume from Checkpoint

```bash
python train_tiered.py --battles 1000 --checkpoint checkpoints/agent_latest.pt
```

### Use GPU (Recommended)

```bash
python train_tiered.py --battles 1000 --device cuda
```

## What You'll See

When training with tier awareness, you'll see output like:

```
[Starting Battle 1]
  Tier: 3 | Format: Doubles
  Bot: Liliana (T00001020_IdolLiliana_ND3.txt)
  Opponent: Rose (T00000737_CowgirlRose_ND3.txt)

[Battle 1] WIN | Tier 3 Doubles | W/L: 1/0 (100.0%)

[Starting Battle 2]
  Tier: 5 | Format: Singles
  Bot: Lady Christy (T00001119_LadyChristy_NS5.txt)
  Opponent: Battle Girl Melinda (T00000412_BattleGirlMelinda_NS5.txt)

[Battle 2] LOSS | Tier 5 Singles | W/L: 1/1 (50.0%)
```

Every 50 battles, you'll see tier-specific statistics:

```
============================================================
Performance by Tier
============================================================
Tier 1:  28W /  22L ( 56.0%) -  50 battles
Tier 2:   5W /   3L ( 62.5%) -   8 battles
Tier 3:   7W /   5L ( 58.3%) -  12 battles
...

============================================================
Performance by Format
============================================================
Singles:  18W /  15L ( 54.5%) -  33 battles
Doubles:  22W /  15L ( 59.5%) -  37 battles
============================================================
```

## Curriculum Learning Strategy

You can train progressively through tiers:

### Beginner (Start Here)
```bash
# Train on Tier 1 first (most teams, easiest matchups)
python train_tiered.py --battles 500 --tier 1

# Then move to Tier 2
python train_tiered.py --battles 300 --tier 2 --checkpoint checkpoints/agent_latest.pt
```

### Intermediate
```bash
# Train on middle tiers
python train_tiered.py --battles 500 --tier 3 --checkpoint checkpoints/agent_latest.pt
python train_tiered.py --battles 500 --tier 4 --checkpoint checkpoints/agent_latest.pt
```

### Advanced
```bash
# Train on harder tiers
python train_tiered.py --battles 500 --tier 5 --checkpoint checkpoints/agent_latest.pt
python train_tiered.py --battles 500 --tier 6 --checkpoint checkpoints/agent_latest.pt
python train_tiered.py --battles 500 --tier 7 --checkpoint checkpoints/agent_latest.pt
```

### Expert (Final Training)
```bash
# Train across all tiers for general performance
python train_tiered.py --battles 2000 --checkpoint checkpoints/agent_latest.pt
```

## Understanding Tiers

In BDSP Battle Tower:
- **Tier 1:** Easier battles, more teams available
- **Tiers 2-6:** Progressive difficulty
- **Tier 7:** Hardest battles, most competitive teams

Training on **all tiers** helps the agent learn diverse strategies.
Training on **specific tiers** helps optimize for that difficulty level.

## Format Differences

### Singles (S)
- 1v1 battles
- More focused strategy
- Prediction is crucial

### Doubles (D)
- 2v2 battles (4 Pokemon teams)
- Team synergy important
- More complex decision space

## Best Practices

### For Quick Testing
```bash
# Small sample on Tier 1
python train_tiered.py --battles 50 --tier 1
```

### For Balanced Training
```bash
# All tiers and formats
python train_tiered.py --battles 2000 --device cuda
```

### For Format Specialization
```bash
# Become a Singles expert
python train_tiered.py --battles 2000 --format singles --device cuda

# Or a Doubles expert
python train_tiered.py --battles 2000 --format doubles --device cuda
```

### For Tier Specialization
```bash
# Master Tier 7 (hardest)
python train_tiered.py --battles 1000 --tier 7 --device cuda
```

## Comparing Performance

After training, compare performance across tiers:

1. **Check logs:**
```bash
# View training history
cat logs/training_history.csv

# View detailed stats (includes tier breakdown)
cat logs/training_log_*.json
```

2. **Evaluate on specific tiers:**
```bash
# You can write custom evaluation scripts to test
# the agent on each tier separately
```

## Statistics Tracking

The tiered trainer saves:
- **Overall win/loss** across all battles
- **Per-tier win/loss** for each of the 7 tiers
- **Per-format win/loss** for Singles and Doubles

This helps you understand:
- Which tiers your agent performs best on
- Whether agent prefers Singles or Doubles
- Where to focus future training

## Example Training Session

Complete workflow:

```bash
# 1. Start server (Terminal 1)
start_server.bat

# 2. Train on Tier 1 to learn basics (Terminal 2)
python train_tiered.py --battles 500 --tier 1 --device cuda

# 3. Expand to all Doubles battles
python train_tiered.py --battles 1000 --format doubles --checkpoint checkpoints/agent_latest.pt --device cuda

# 4. Train across all tiers for generalization
python train_tiered.py --battles 2000 --checkpoint checkpoints/agent_latest.pt --device cuda

# 5. Specialize on Tier 7
python train_tiered.py --battles 500 --tier 7 --checkpoint checkpoints/agent_latest.pt --device cuda
```

## Key Features

✅ **Fair Matchmaking** - Only battles teams of same tier and format
✅ **Tier Statistics** - Track performance by tier
✅ **Format Statistics** - Track Singles vs Doubles separately
✅ **Flexible Training** - Train on all tiers or specialize
✅ **Curriculum Learning** - Progress through difficulties
✅ **Full Metadata** - Every team has trainer name, class, tier, format

## Quick Reference

| Command | Purpose |
|---------|---------|
| `--battles N` | Number of battles |
| `--tier 1-7` | Specific tier only |
| `--format S/D` | Singles or Doubles only |
| `--checkpoint PATH` | Resume from saved model |
| `--device cuda` | Use GPU |
| `--bot-name NAME` | Custom bot name |

## Summary

You now have **tier-aware training** that ensures fair battles between teams of matching difficulty and format. This provides:

1. **Better learning** - Agent faces appropriate difficulty
2. **Detailed statistics** - Understand performance by tier/format
3. **Flexible training** - Specialize or generalize as needed
4. **Fair evaluation** - Compare teams within their tier

Happy training! 🚀
