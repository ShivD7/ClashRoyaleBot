# ClashRoyaleBot developer guide

This repository is a deterministic, Clash Royale-style battle simulator written
in Python. Pygame supplies the window, input, drawing, rectangles, and vector
math. The project does **not** yet contain a trainable reinforcement-learning
environment; it contains the simulator and the controller boundary that the RL
environment will eventually use.

This guide explains the current code from the outside in. Read it before trying
to understand individual drawing or collision functions.

## Quick start

```bash
cd /Users/shivdesai/Desktop/ClashRoyaleBot
source .venv/bin/activate
python arena_viewer.py
```

Run the tests with:

```bash
python -m pytest -q
```

The default match is a human blue player against the scripted red controller.
You can replace either controller from the command line:

```bash
python arena_viewer.py --blue-controller random --red-controller fixed
```

Valid controller names are `human`, `random`, `scripted`, `fixed`, and `rl`.
The current `rl` controller waits forever unless training code injects a policy.

## Main modules

| File | Responsibility | Does not own |
| --- | --- | --- |
| `arena_viewer.py` | Match coordination, decks, hands, Elixir, legal placement, clocks, input, and rendering | Detailed troop combat |
| `battle_engine.py` | Towers, troops, buildings, targeting, movement, collisions, attacks, spells, health, and death | Pygame events, HUDs, decks, or Elixir |
| `controllers.py` | A safe interface for human, random, scripted, fixed, and future learned decision makers | Authority to mutate the match |
| `training_interface.py` | Numerical observations, discrete action IDs, and legality masks | Episode timing or rewards |
| `training_env.py` | Gymnasium reset/step lifecycle, fixed observation arrays, opponent scheduling, and rewards | Combat or placement rules |

The dependency direction is intentional:

```text
controllers.py        battle_engine.py
       \                  /
        \                /
             arena_viewer.py
                    |
               Pygame window
```

`controllers.py` and `battle_engine.py` do not import `arena_viewer.py`. This
keeps the combat rules reusable by tests and by a future headless RL wrapper.

## State ownership

Understanding who is allowed to change each value is the most important mental
model in the project.

| State | Owner |
| --- | --- |
| Unit positions, health, targets, cooldowns, projectiles | `BattleEngine` |
| Crown score derived from destroyed towers | `BattleEngine` |
| A player's deck, four-card hand, queue, and Elixir | `PlayerState` inside `ArenaViewer` |
| Regulation, overtime, tiebreaker, and match result | `ArenaViewer` |
| Mouse selection, drag state, fonts, and animation-only effects | `ArenaViewer` |
| A controller's private strategy memory | That controller |

A controller is never trusted with live mutable objects. It receives a frozen
`ControllerContext`, returns a requested `PlayCardAction`, and the viewer either
accepts or rejects that request using the authoritative game rules.

## Coordinate systems and units

The arena is an 18-column by 32-row grid. One tile is 25 logical pixels.

There are three related coordinate representations:

1. **Tile coordinates** are `(column, row)` and are used for card actions.
2. **Logical pixels** are used by combat entities and all drawing code. The
   arena is first rendered at its full logical size.
3. **Window pixels** are the scaled desktop-window coordinates reported by the
   mouse. They are converted back to logical pixels before hit testing.

The top of the arena belongs to red and the bottom belongs to blue. Increasing
`x` moves right and increasing `y` moves down.

Combat statistics use mixed units deliberately:

- health and damage are hit points;
- `hit_speed`, spell travel time, and status durations are seconds;
- attack, sight, splash, and spell radii are measured in tiles;
- immutable card movement/projectile speeds are tile-based rates;
- runtime entity positions, body radii, and velocities use logical pixels.

`BattleEngine` performs the tile-to-pixel conversions at the boundary where a
card template becomes a live entity.

## The complete journey of one card action

Every human or computer play eventually follows the same path:

```text
mouse/controller
    -> PlayCardAction(hand_slot, tile)
    -> ArenaViewer.try_play_action(...)
    -> check match phase and hand slot
    -> check terrain and live entity footprints
    -> check and spend Elixir
    -> BattleEngine.deploy_card(...)
    -> spawn entities or schedule a spell
    -> rotate the played card through CardCycle
```

The order matters. An invalid placement must not spend Elixir or cycle the
hand. The tests explicitly protect that transaction-like behavior.

Spells and troops split inside `BattleEngine.deploy_card`:

- troop/building cards create one or more `BattleEntity` objects;
- instant spells apply immediately in simulation time;
- travelling spells create a `PendingSpell` and land after a deterministic
  countdown;
- the Fireball object drawn by `ArenaViewer` is visual only and cannot change
  when damage occurs.

## The simulation clock

Rendering and gameplay use different clocks.

- The window attempts to draw at 60 frames per second.
- Gameplay advances only in exact 50 ms steps: 20 simulation ticks per second.
- Slow render frames may execute several simulation ticks to catch up.
- Fast render frames save their unused milliseconds for the next frame.

This fixed timestep makes test results and future RL episodes reproducible. A
headless trainer will call the same simulation updates without drawing frames.

Within one `BattleEngine.update(delta_seconds)` call, the important order is:

1. Activate King Towers whose activation condition is already true.
2. Age deployed buildings and generate due spawner waves.
3. Reduce status durations and attack cooldowns.
4. Validate or acquire each entity's target.
5. Resolve ready attacks and plan movement from the shared starting positions.
6. Apply planned movement and resolve circular-body overlaps.
7. Advance projectiles and resolve their impacts.
8. Land travelling spells whose timers reached zero.
9. Recheck King Tower activation after all damage.

Do not casually reorder these stages. Their order is part of the simulator's
rules and determinism.

## Card templates versus live entities

`Card`, `UnitStats`, `SpellStats`, and `SpawnerStats` are immutable definitions.
They describe what should be created but never store current health or position.

`BattleEntity`, `Projectile`, and `PendingSpell` are mutable episode objects.
They belong to one battle and change on every simulation tick.

For example, every Skeleton in a Skeleton Army shares the same immutable
`UnitStats`, but each spawned Skeleton gets a different entity ID, position,
health value, target, and cooldown.

## Ground movement and bridges

Flying units can move directly toward targets. Ground units cannot cross open
water, so the engine gives them intermediate bridge waypoints.

A ground unit:

1. chooses a bridge using travel distance plus a congestion penalty;
2. remembers that lane in `lane_x` so it does not oscillate between bridges;
3. steers around nearby bodies;
4. moves simultaneously with the other planned movers;
5. participates in mass-weighted collision separation;
6. is constrained back onto valid grass or bridge geometry when necessary.

Hog Rider has a special outer-edge river jump. Flying entities and jumping Hog
Riders occupy different targeting/collision conditions from ordinary ground
troops.

## Match phases

The match coordinator progresses through these states:

```text
regulation (3:00)
    -> crown leader wins, or tied score enters overtime
overtime (2:00 sudden death)
    -> first crown lead wins, or tied score enters tiebreaker
tiebreaker
    -> normal combat stops and all living Crown Towers drain equally
finished
    -> simulation freezes until rematch or return to home screen
```

Destroying a King Tower immediately wins. Destroying a Princess Tower awards a
crown, activates its allied King Tower, and unlocks forward deployment in that
lane.

## Controllers and future reinforcement learning

`Match` now exposes a renderer-independent numerical training boundary:

- `match.observation(team)` reports the observing side's hand, card costs, and
  Elixir; public tower and battlefield state; scores, time, phase, projectiles,
  and pending spells.
- Opponent hand, card queue, and Elixir are intentionally private and do not
  appear in observations. Deployed opponent objects are public.
- `match.action_mask(team)` returns 2,305 flags. Index 0 is wait; indices
  1–2,304 map four hand slots across the arena's 576 tiles.
- `match.apply_discrete_action(team, action_id)` uses that fixed mapping and the
  same placement validator as the Pygame viewer.

`ClashRoyaleEnv` wraps that boundary in Gymnasium's single-agent API. Blue is
the learning player and red uses the existing scripted controller:

```python
from training_env import ClashRoyaleEnv

env = ClashRoyaleEnv()
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(action)
```

Each call to `step` advances five shared 0.05-second physics updates, or 0.25
simulated seconds. The scripted opponent acts through the match's existing
controller schedule during those same updates. `info["action_mask"]` and
`env.action_masks()` expose the current mask for MaskablePPO-style trainers.
The initial reward is sparse: +1 for a blue win, -1 for a red win, and 0 for a
draw or a non-terminal decision.

The reward values are named constants in `training_env.py` and are awarded only
from the terminal winner. Playing cards, spending Elixir, ordinary tower
damage, and automatic tiebreaker damage do not create intermediate reward.
This prevents the learner from maximizing a convenient activity counter rather
than learning to win. If sparse learning proves too slow, tower-damage shaping
can be added later with an explicit exclusion for automatic tiebreaker drain.

Variable battlefield collections are padded into fixed-size numerical arrays
with presence masks so vectorized trainers can batch observations. Any count
above a configured capacity is reported in `info["observation_overflow"]`.

## Baseline training and machine support

Check the current interpreter before installing or training:

```bash
python training_baseline.py check
```

The baseline uses `sb3-contrib` MaskablePPO with `MultiInputPolicy` and separate
64-by-64 feed-forward policy/value networks. It trains blue against the fixed
scripted opponent, saves the checkpoint, reloads it immediately, and makes a
masked prediction to verify the saved artifact:

```bash
python training_baseline.py train --timesteps 10000
```

By default, two held-out episodes per opponent/side are run before and after
training on the same evaluation seeds. The comparison is written to
`checkpoints/maskable_ppo_baseline.evaluation.json`; increase confidence with
`--eval-episodes 10` once the smoke run succeeds. These evaluation matches are
not inserted into PPO's training buffer.

Evaluation seeds are separate from the training seed. Evaluation reports wins,
losses, draws, and win rate against random and scripted opponents from both the
blue and red sides:

```bash
python training_baseline.py evaluate --episodes 10
```

To load the same model through the existing `RLController` and watch it in the
Pygame viewer:

```bash
python training_baseline.py watch --team blue --opponent scripted
```

Install `requirements-training.txt` on a currently supported PyTorch machine.
PyTorch's current binary matrix supports Linux x86_64, Windows x86_64, and
Apple-Silicon macOS, but not Intel macOS. The checked MacBook Air is x86_64 with
two CPU cores, 8 GB memory, Intel UHD 617 graphics, and Python 3.14; its normal
virtual environment can run the simulator but cannot install current PyTorch.

The recommended training machine is the Dell XPS with 64-bit Linux or Windows
and Python 3.11 or 3.12:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux:   source .venv/bin/activate
python -m pip install -r requirements-training.txt
python training_baseline.py check
python training_baseline.py train --timesteps 10000 --device auto
```

If the Dell has a supported NVIDIA GPU, install the PyTorch build selected for
its CUDA driver from pytorch.org before installing the remaining requirements.
Otherwise, MaskablePPO runs on its CPU with `--device cpu`.

`environment-intel-mac.yml` provides an optional legacy Intel-Mac CPU fallback
using Python 3.10 and conda-forge PyTorch 2.5.1. It is expected to be slow and
is outside PyTorch's current tested binary matrix:

```bash
conda env create -f environment-intel-mac.yml
conda activate clash-royale-intel-mac
python training_baseline.py check
python training_baseline.py train --timesteps 1000 --device cpu
```

The remaining training work is to add:

- optional intermediate reward shaping;
- randomized decks and opponents driven by the Gymnasium seed;
- experiment configuration and metrics;
- self-play after the single-agent baseline is stable.

## Tests as executable documentation

The tests are grouped by gameplay contract rather than by private method. Read
them alongside the implementation:

- `tests/test_battle_engine.py` documents targeting, movement, combat, status
  effects, buildings, projectiles, tower activation, and scoring.
- `tests/test_arena_viewer.py` documents geometry, card data, placement, decks,
  Elixir, match phases, resets, and input-facing behavior.
- `tests/test_controllers.py` documents the controller boundary, independent
  player state, legal actions, decision scheduling, and the RL adapter.
- `tests/test_training_interface.py` documents observation visibility, action
  encoding, masks, and window-free match completion.
- `tests/test_training_env.py` documents Gymnasium reset/step behavior, episode
  timing, truncation, reward signs, draw handling, tiebreaker behavior, seeded
  trajectory reproduction, full-match termination, and API compliance.

Most tests follow Arrange–Act–Assert:

1. arrange the smallest possible battle state;
2. advance an exact amount of simulation time or request one action;
3. assert the public behavior that must remain stable.

When changing a rule, add or update the focused test before changing broad
integration behavior.

## Dependency note for Intel Macs

The current simulator and tests only import `pygame-ce` and `pytest`. The main
requirements file also declares the planned modern RL stack. PyTorch no longer
publishes recent native macOS x86_64 wheels, so `torch>=2.8` cannot be installed
natively on an Intel Mac. Simulator and environment development can still be
done locally; modern neural-network training should run on a supported Linux or
Apple Silicon machine.
