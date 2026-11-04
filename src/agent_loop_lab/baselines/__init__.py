"""Baselines for M4 (spec v0.2 §9.5): Phase C supervised training, rules engine, constant priors.

This package trains with ``torch.optim``. It must never be imported by any
weight-build path (runner, packet, ``weight_program/``); nothing there
imports it, and ``tests/test_m4_phase_c.py`` checks that.
"""
