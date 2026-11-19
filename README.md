# ClashRoyaleBot

A local Clash Royale-style simulator built with Python and Pygame.

* Builds eight-card decks from the card collection.
* Plays cards by clicking, dragging, or using keyboard shortcuts.
* Tracks card rotation, Elixir costs, and changing Elixir generation.
* Simulates troops, buildings, towers, projectiles, and spells.
* Handles targeting, bridge movement, flying units, river jumps, and collisions.
* Includes splash damage, knockback, stun, freeze, and spawning buildings.
* Enforces placement rules and unlocks territory after tower destruction.
* Tracks crowns, regulation time, overtime, and tiebreakers.
* Supports rematches and returning to the deck screen.
* Runs human, random, scripted, fixed, or learned controllers.
* Runs matches without a window using a fixed simulation timestep.
* Provides a Gymnasium environment with numerical observations, legal action masks, and rewards.
* Trains MaskablePPO policies, saves checkpoints, and resumes training.
* Evaluates saved policies against random and scripted opponents from either side.
* Loads trained policies in the Pygame viewer.

This is a standalone simulator. It does not control the live Clash Royale game.
