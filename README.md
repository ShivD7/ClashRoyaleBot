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

## The three main modules

| File | Responsibility | Does not own |
| --- | --- | --- |
| `arena_viewer.py` | Match coordination, decks, hands, Elixir, legal placement, clocks, input, and rendering | Detailed troop combat |
| `battle_engine.py` | Towers, troops, buildings, targeting, movement, collisions, attacks, spells, health, and death | Pygame events, HUDs, decks, or Elixir |
| `controllers.py` | A safe interface for human, random, scripted, fixed, and future learned decision makers | Authority to mutate the match |

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

The current `RLController` is only an adapter around a callable policy. It is
not yet a Gymnasium environment and its `ControllerContext` does not yet include
full battlefield observations.

The recommended next architectural step is to extract the non-visual match
state from `ArenaViewer` into a headless match class, then add:

- `reset(seed=...)` and `step(action)`;
- a fixed numerical observation representation;
- a discrete action encoding plus a legal-action mask;
- terminal and intermediate rewards;
- deterministic episode seeding;
- scripted-opponent and self-play modes.

The existing action space naturally maps four hand slots across 576 arena
tiles. A future discrete encoding can reserve action zero for “wait” and use
the remaining 2,304 values for `(hand slot, row, column)`.

## Tests as executable documentation

The tests are grouped by gameplay contract rather than by private method. Read
them alongside the implementation:

- `tests/test_battle_engine.py` documents targeting, movement, combat, status
  effects, buildings, projectiles, tower activation, and scoring.
- `tests/test_arena_viewer.py` documents geometry, card data, placement, decks,
  Elixir, match phases, resets, and input-facing behavior.
- `tests/test_controllers.py` documents the controller boundary, independent
  player state, legal actions, decision scheduling, and the RL adapter.

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
