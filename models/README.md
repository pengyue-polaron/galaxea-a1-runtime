# Local Model Registry

`models/` is the only path that tracked inference configs should use for model
weights. Everything in this directory except this file is ignored by Git.

The local layout is:

```text
models/
  base/lingbot-va-base
  checkpoints/lingbot/a1_banana_in_plate/checkpoint_step_500
  checkpoints/lingbot/a1_banana_in_plate/checkpoint_step_1000
  checkpoints/act/a1_banana_joint_state_30k/checkpoint_step_30000
  runtime/lingbot/a1_banana_in_plate/checkpoint_step_500
```

The first three paths may be symlinks to downloaded models or native training
outputs. `runtime/` is disposable and is assembled by the LingBot launcher from
the registered base and checkpoint components.

Register the current supported slots without copying their contents:

```bash
just model-link lingbot-base /path/to/lingbot-va-base
just model-link lingbot-a1-banana-step500 /path/to/checkpoint_step_500
just model-link lingbot-a1-banana-step1000 /path/to/checkpoint_step_1000
just model-link act-a1-banana-step30000 /path/to/pretrained_model
just models
```

Storage ownership is intentionally separate:

- `models/`: canonical deployment references and generated model assemblies.
- `train_out/` and `outputs/train/`: native training outputs.
- `outputs/`: inference logs, observations, reviews, and evaluations.
- `data/`: raw and converted datasets.
- `.cache/`: disposable package/runtime caches, never canonical weights.

Do not commit weights and do not add Git LFS to this repository. `just models`
fails when a configured model is missing, a file over 100 MiB is tracked, or
Git reports garbage left by an interrupted pack operation.
