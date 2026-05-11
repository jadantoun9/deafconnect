# Configs

YAML per-track and per-evaluation. Read by `train.py` and `eval.py`.

Conventions:
- One file per *recipe* (architecture + hyperparameters + dataset). Don't share configs
  across recipes — duplicate and edit, even if it means a few repeated keys.
- Comments explain *why*, not *what*. Keep defaults sensible; comment any unusual value.
- Configs are loaded with `pyyaml` — no Hydra, no OmegaConf, no interpolation. If a knob
  needs to vary by sweep, expose it as a CLI flag in `train.py`, not a config interpolation.
