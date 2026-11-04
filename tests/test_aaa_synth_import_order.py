"""Sorts before the guard tests and imports the generator package, to prove the
conftest isolation keeps guard tests working when synth is already imported."""

from agent_loop_lab.synth import labeller, latent  # noqa: F401


def test_synth_importable():
    assert latent.LatentStep and labeller.label_session
