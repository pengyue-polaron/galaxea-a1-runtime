# Local Model Registry

`models/` is the only path that tracked inference configs should use for model
weights. Everything in this directory except this file is ignored by Git.

The local layout is:

```text
models/
  base/lingbot-va-base
  checkpoints/lingbot/a1_agentview_square/latest
  checkpoints/act/a1_agentview_square/latest
  runtime/lingbot/a1_agentview_square/latest
```

The base and checkpoint paths may be symlinks to downloaded models or native
training outputs. `runtime/` is disposable and is assembled by the LingBot launcher from
the registered base and checkpoint components.

Register the current supported slots without copying their contents:

```bash
just model-link lingbot-base /path/to/lingbot-va-base
just model-link lingbot-a1-agentview-square /path/to/new_lingbot_checkpoint
just model-link act-a1-agentview-square /path/to/new_act_pretrained_model
just models
```

Both deployment checkpoints must be trained from data whose AgentView input is
the configured `(x=103, y=0, width=480, height=480)` crop. ACT additionally
stores that raw front shape in its checkpoint input-feature contract and will
refuse to load a mismatched checkpoint.

After registering new weights, update the LingBot prompt and q01/q99 statistics
from the same training run, then set `deployment_ready = true` in each reviewed
inference config. Both profiles remain dry-run until execution is enabled
separately.

Storage ownership is intentionally separate:

- `models/`: canonical deployment references and generated model assemblies.
- `train_out/` and `outputs/train/`: native training outputs.
- `outputs/`: inference logs, observations, reviews, and evaluations.
- `data/`: raw and converted datasets.
- `.cache/`: disposable package/runtime caches, never canonical weights.

Do not commit weights and do not add Git LFS to this repository. `just models`
fails when a configured model is missing, a file over 100 MiB is tracked, or
Git reports garbage left by an interrupted pack operation.
