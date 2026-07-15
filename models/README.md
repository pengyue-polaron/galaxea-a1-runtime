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

The base and checkpoint paths may be symlinks to weights produced or downloaded
elsewhere. This checkout does not train models. `runtime/` is disposable and is
assembled by the LingBot launcher from the registered base and checkpoint
components.

Register the current supported slots without copying their contents:

```bash
just model-link lingbot-base /path/to/lingbot-va-base
just model-link lingbot-a1-agentview-square /path/to/new_lingbot_checkpoint
just model-link act-a1-agentview-square /path/to/new_act_pretrained_model
just models
```

Both deployment checkpoints must match the camera feature contract derived
from their referenced System config. ACT also stores the front image shape in
its checkpoint contract and refuses to load a mismatch.

After registering new weights, update the LingBot prompt, expected weight size,
and q01/q99 statistics from that same training run before setting LingBot
`deployment_ready = true`. Review the ACT checkpoint contract separately before
setting ACT `deployment_ready = true`. Both deployments remain dry-run until
their independent execution setting is enabled.

There is no local training-output root. Bring a reviewed checkpoint onto this
machine, register it with `just model-link`, and keep tracked deployment configs
pointing only at the resulting `models/` slot.

Do not commit weights and do not add Git LFS to this repository. `just models`
fails when a configured model is missing, a file over 100 MiB is tracked, or
Git reports garbage left by an interrupted pack operation.
