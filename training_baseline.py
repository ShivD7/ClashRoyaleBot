"""Train, save, evaluate, and watch a small MaskablePPO baseline.

PyTorch and SB3 imports are intentionally lazy. The simulator and its tests can
therefore run on machines that cannot install the training stack, while saved
models remain portable to any supported CPU or GPU machine.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from arena_viewer import ArenaViewer, Match
from controllers import (
    HumanController,
    PlayCardAction,
    PlayerController,
    RandomController,
    RLController,
    ScriptedController,
)
from training_env import ClashRoyaleEnv, encode_fixed_observation
from training_interface import action_mask, decode_action

DEFAULT_MODEL_PATH = Path("checkpoints/maskable_ppo_baseline.zip")
DEFAULT_TIMESTEPS = 10_000


@dataclass(frozen=True)
class CompatibilityReport:
    supported: bool
    system: str
    machine: str
    python: str
    torch_installed: bool
    sb3_contrib_installed: bool
    device: str | None
    reason: str


@dataclass(frozen=True)
class EvaluationResult:
    opponent: str
    learner_team: str
    episodes: int
    wins: int
    losses: int
    draws: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.episodes if self.episodes else 0.0


def compatibility_report() -> CompatibilityReport:
    """Inspect whether the current interpreter can run the PPO stack."""
    system = platform.system()
    machine = platform.machine()
    python_version = platform.python_version()
    torch_installed = importlib.util.find_spec("torch") is not None
    sb3_installed = importlib.util.find_spec("sb3_contrib") is not None
    device = None

    if torch_installed:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    intel_mac = system == "Darwin" and machine == "x86_64"
    if intel_mac and not torch_installed:
        reason = (
            "Current PyTorch releases do not publish macOS x86_64 wheels. "
            "Use the documented legacy Conda CPU environment or train on a "
            "supported Linux/Windows x86_64 machine."
        )
    elif not torch_installed:
        reason = "PyTorch is not installed in this interpreter."
    elif not sb3_installed:
        reason = "sb3-contrib is not installed in this interpreter."
    else:
        reason = f"Training dependencies are available on {device}."

    return CompatibilityReport(
        supported=torch_installed and sb3_installed,
        system=system,
        machine=machine,
        python=python_version,
        torch_installed=torch_installed,
        sb3_contrib_installed=sb3_installed,
        device=device,
        reason=reason,
    )


def _maskable_ppo() -> Any:
    report = compatibility_report()
    if not report.supported:
        raise RuntimeError(report.reason)
    from sb3_contrib import MaskablePPO

    return MaskablePPO


def _saved_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.suffix == ".zip" else resolved.with_suffix(".zip")


def train_baseline(
    *,
    total_timesteps: int = DEFAULT_TIMESTEPS,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    resume_from: str | Path | None = None,
    seed: int = 42,
    device: str = "auto",
    verbose: int = 1,
    evaluation_episodes: int = 0,
) -> Path:
    """Train or resume a masked PPO policy, then save and reload it."""
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    resume_path = _saved_path(resume_from) if resume_from is not None else None
    if resume_path is not None and not resume_path.exists():
        raise FileNotFoundError(f"No trained model found at {resume_path}")

    MaskablePPO = _maskable_ppo()
    env = ClashRoyaleEnv(learner_team="blue", opponent="scripted")
    model = (
        MaskablePPO.load(resume_path, env=env, device=device)
        if resume_path is not None
        else MaskablePPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            seed=seed,
            device=device,
            verbose=verbose,
            policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        )
    )
    model.verbose = verbose
    before = (
        evaluate_policy(
            model,
            episodes_per_matchup=evaluation_episodes,
            seed=seed + 10_000,
        )
        if evaluation_episodes
        else ()
    )
    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=resume_path is None,
    )

    destination = _saved_path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(destination)

    # Reload immediately and make one masked prediction. A corrupt or
    # incompatible checkpoint therefore fails during training, not days later.
    loaded = MaskablePPO.load(destination, env=env, device=device)
    observation, info = env.reset(seed=seed)
    loaded.predict(
        observation,
        deterministic=True,
        action_masks=info["action_mask"],
    )
    env.close()
    if evaluation_episodes:
        after = evaluate_policy(
            loaded,
            episodes_per_matchup=evaluation_episodes,
            seed=seed + 10_000,
        )
        comparison_path = destination.with_suffix(".evaluation.json")
        comparison_path.write_text(
            json.dumps(
                {
                    "evaluation_seed": seed + 10_000,
                    "episodes_per_matchup": evaluation_episodes,
                    "before_training": [_result_dict(result) for result in before],
                    "after_training": [_result_dict(result) for result in after],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return destination


def _load_model(model_path: str | Path, device: str = "auto") -> Any:
    MaskablePPO = _maskable_ppo()
    path = _saved_path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"No trained model found at {path}")
    return MaskablePPO.load(path, device=device)


def _predict_action(
    model: Any,
    observation: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    deterministic: bool = True,
) -> int:
    action, _ = model.predict(
        observation,
        deterministic=deterministic,
        action_masks=mask,
    )
    return int(np.asarray(action).item())


def attach_model_to_rl_controller(
    match: Match,
    team: str,
    model: Any,
    *,
    deterministic: bool = True,
) -> RLController:
    """Attach a loaded numerical policy through the existing RLController."""

    def policy(_context: object) -> PlayCardAction | None:
        observation, _ = encode_fixed_observation(match, team)
        mask = np.asarray(action_mask(match, team), dtype=np.bool_)
        action_id = _predict_action(
            model,
            observation,
            mask,
            deterministic=deterministic,
        )
        return decode_action(action_id)

    controller = RLController(policy)
    match.controllers[team] = controller
    return controller


def evaluate_policy(
    model: Any,
    *,
    episodes_per_matchup: int = 10,
    seed: int = 10_000,
) -> tuple[EvaluationResult, ...]:
    """Measure one policy on held-out matches from both arena sides."""
    if episodes_per_matchup <= 0:
        raise ValueError("episodes_per_matchup must be positive")
    results = []
    matchup_index = 0
    for opponent in ("random", "scripted"):
        for learner_team in ("blue", "red"):
            wins = losses = draws = 0
            env = ClashRoyaleEnv(
                learner_team=learner_team,
                opponent=opponent,
            )
            for episode in range(episodes_per_matchup):
                episode_seed = seed + matchup_index * 10_000 + episode
                observation, info = env.reset(seed=episode_seed)
                terminated = truncated = False
                while not (terminated or truncated):
                    action_id = _predict_action(
                        model,
                        observation,
                        info["action_mask"],
                    )
                    observation, _, terminated, truncated, info = env.step(
                        action_id,
                    )
                if info["winner"] == learner_team:
                    wins += 1
                elif info["winner"] is None:
                    draws += 1
                else:
                    losses += 1
            env.close()
            results.append(
                EvaluationResult(
                    opponent=opponent,
                    learner_team=learner_team,
                    episodes=episodes_per_matchup,
                    wins=wins,
                    losses=losses,
                    draws=draws,
                )
            )
            matchup_index += 1
    return tuple(results)


def evaluate_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    episodes_per_matchup: int = 10,
    seed: int = 10_000,
    device: str = "auto",
) -> tuple[EvaluationResult, ...]:
    """Load a checkpoint and evaluate it on matches not used for training."""
    model = _load_model(model_path, device=device)
    return evaluate_policy(
        model,
        episodes_per_matchup=episodes_per_matchup,
        seed=seed,
    )


def _opponent_controller(name: str, seed: int) -> PlayerController:
    if name == "scripted":
        return ScriptedController()
    if name == "random":
        return RandomController(seed=seed)
    raise ValueError("opponent must be 'scripted' or 'random'")


def watch_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    learner_team: str = "blue",
    opponent: str = "scripted",
    seed: int = 42,
    device: str = "auto",
) -> None:
    """Open the Pygame viewer and let a loaded model play via RLController."""
    if learner_team not in {"blue", "red"}:
        raise ValueError("learner_team must be 'blue' or 'red'")
    opponent_team = "red" if learner_team == "blue" else "blue"
    controllers: dict[str, PlayerController] = {
        learner_team: HumanController(),
        opponent_team: _opponent_controller(opponent, seed),
    }
    viewer = ArenaViewer(
        blue_controller=controllers["blue"],
        red_controller=controllers["red"],
    )
    model = _load_model(model_path, device=device)
    attach_model_to_rl_controller(viewer, learner_team, model)
    viewer.start_match()
    viewer.run()


def _print_evaluation(results: tuple[EvaluationResult, ...]) -> None:
    for result in results:
        print(json.dumps(_result_dict(result), sort_keys=True))


def _result_dict(result: EvaluationResult) -> dict[str, object]:
    values: dict[str, object] = asdict(result)
    values["win_rate"] = result.win_rate
    return values


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate MaskablePPO")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="check this machine's training support")

    train_parser = subparsers.add_parser("train", help="train and save a baseline")
    train_parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    train_parser.add_argument(
        "--resume",
        type=Path,
        help="continue training from an existing MaskablePPO checkpoint",
    )
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument(
        "--eval-episodes",
        type=int,
        default=2,
        help="held-out episodes per opponent/side before and after training",
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="run held-out matches")
    evaluate_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    evaluate_parser.add_argument("--episodes", type=int, default=10)
    evaluate_parser.add_argument("--seed", type=int, default=10_000)
    evaluate_parser.add_argument("--device", default="auto")

    watch_parser = subparsers.add_parser("watch", help="watch a loaded model play")
    watch_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    watch_parser.add_argument("--team", choices=("blue", "red"), default="blue")
    watch_parser.add_argument(
        "--opponent",
        choices=("random", "scripted"),
        default="scripted",
    )
    watch_parser.add_argument("--seed", type=int, default=42)
    watch_parser.add_argument("--device", default="auto")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> None:
    settings = parse_arguments(arguments)
    try:
        if settings.command == "check":
            print(json.dumps(asdict(compatibility_report()), indent=2, sort_keys=True))
        elif settings.command == "train":
            saved = train_baseline(
                total_timesteps=settings.timesteps,
                model_path=settings.model,
                resume_from=settings.resume,
                seed=settings.seed,
                device=settings.device,
                evaluation_episodes=settings.eval_episodes,
            )
            print(f"Saved and reloaded model: {saved}")
        elif settings.command == "evaluate":
            _print_evaluation(
                evaluate_model(
                    settings.model,
                    episodes_per_matchup=settings.episodes,
                    seed=settings.seed,
                    device=settings.device,
                )
            )
        else:
            watch_model(
                settings.model,
                learner_team=settings.team,
                opponent=settings.opponent,
                seed=settings.seed,
                device=settings.device,
            )
    except (FileNotFoundError, RuntimeError) as error:
        raise SystemExit(f"Training command unavailable: {error}") from None


if __name__ == "__main__":
    main()
