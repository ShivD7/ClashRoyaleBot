"""Contracts for the portable MaskablePPO training and playback helpers."""

import numpy as np
import pytest

from arena_viewer import Match
from controllers import HumanController, ScriptedController
from training_baseline import (
    CompatibilityReport,
    EvaluationResult,
    attach_model_to_rl_controller,
    compatibility_report,
    parse_arguments,
    train_baseline,
)


class FirstLegalModel:
    """Small prediction double with the same method used by MaskablePPO."""

    def __init__(self) -> None:
        self.call_count = 0

    def predict(
        self,
        observation: dict[str, np.ndarray],
        *,
        deterministic: bool,
        action_masks: np.ndarray,
    ) -> tuple[np.ndarray, None]:
        assert observation["hand_card_ids"].shape == (4,)
        assert deterministic
        self.call_count += 1
        legal_placements = np.flatnonzero(action_masks[1:]) + 1
        action = legal_placements[0] if len(legal_placements) else 0
        return np.asarray(action), None


def test_loaded_policy_plays_through_existing_rl_controller() -> None:
    match = Match(
        blue_controller=HumanController(),
        red_controller=ScriptedController(),
    )
    model = FirstLegalModel()

    controller = attach_model_to_rl_controller(match, "blue", model)
    match.advance(5)

    assert match.controllers["blue"] is controller
    assert model.call_count == 1
    assert any(deployment for deployment in match.deployments)


def test_compatibility_report_is_actionable() -> None:
    report = compatibility_report()

    assert report.system
    assert report.machine
    assert report.python
    assert report.reason
    assert report.supported == (report.torch_installed and report.sb3_contrib_installed)


def test_missing_training_stack_fails_with_report_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = CompatibilityReport(
        supported=False,
        system="ExampleOS",
        machine="x86_64",
        python="3.12",
        torch_installed=False,
        sb3_contrib_installed=False,
        device=None,
        reason="Install the training dependencies.",
    )
    monkeypatch.setattr("training_baseline.compatibility_report", lambda: report)

    with pytest.raises(RuntimeError, match=report.reason):
        train_baseline(total_timesteps=1)


def test_evaluation_result_reports_win_rate() -> None:
    result = EvaluationResult(
        opponent="scripted",
        learner_team="red",
        episodes=10,
        wins=3,
        losses=6,
        draws=1,
    )

    assert result.win_rate == 0.3


def test_train_arguments_support_resuming_to_a_new_checkpoint() -> None:
    settings = parse_arguments(
        [
            "train",
            "--timesteps",
            "50000",
            "--resume",
            "checkpoints/starting.zip",
            "--model",
            "checkpoints/extended.zip",
        ],
    )

    assert settings.timesteps == 50_000
    assert settings.resume.name == "starting.zip"
    assert settings.model.name == "extended.zip"


def test_missing_resume_checkpoint_fails_before_training(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="No trained model found"):
        train_baseline(
            total_timesteps=1,
            resume_from=tmp_path / "missing.zip",
        )
