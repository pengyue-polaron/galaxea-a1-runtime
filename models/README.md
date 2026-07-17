# Local Model Registry

`models/` contains ignored deployment weights. Tracked identity and behavior
live under `configs/models/`, not beside downloaded files.

## Ownership

Inference is composed from five independently reusable layers:

| Layer | Owns |
| --- | --- |
| System | cameras, ROS topics, physical limits, and relay safety |
| Backend | exact code revision, dependency lock, environment, and engine |
| Model | immutable source revision, checkpoint step, complete content manifest, and weight-specific contract |
| Task catalog | approved runtime prompts plus explicit train/OOD provenance |
| Deployment | references to the other layers, task catalog, server lifecycle, and execution choices |

A model's local root is always derived from its tracked identity:

```text
models/artifacts/<model-id>/<40-character-source-revision>/
```

There is no `latest` alias for managed artifacts and no deployment-owned weight
path. Adding another checkpoint means adding a new model descriptor, manifest,
and contract; existing deployments can then reference it without changing a
backend. Multiple tasks and model families can coexist without link farms or
copied paths.

## Integrity and publication

Every managed model pins an immutable Hugging Face commit. Its tracked manifest
lists the exact non-cache file set, byte size, and SHA-256 of every file.
Download occurs in a hidden sibling staging directory. Only after full
validation does an atomic rename expose the final revision directory. A crash
leftover intentionally blocks reuse until it is inspected.

Fetch or verify one descriptor directly:

```bash
just model-fetch configs/models/pi05/fruit_placement_eef.toml
just model-verify configs/models/pi05/fruit_placement_eef.toml
```

Validate every configured model:

```bash
just models
```

## Managed EEF policies

The configured LingBot and OpenPI pi0.5 models use separate pinned source trees
and dependency environments, but the same model-store and service-contract
boundaries:

```bash
just lingbot-setup
just lingbot-verify
just lingbot-smoke

just pi05-setup
just pi05-verify
just pi05-smoke
```

Setup is hardware-free: it verifies the backend checkout and lock, synchronizes
the backend-local environment, fetches the exact model revision, and validates
all artifact hashes. Smoke starts only the GPU policy server and sends synthetic
camera/state inputs. It does not initialize ROS, open cameras, or publish robot
commands.

The service and client exchange an exact startup handshake. It covers source and
model revisions, manifest digest, the complete task catalog, camera keys and shapes, state/action
layout, normalization, coordinate mode, and engine settings. A mismatch fails
before any action can be accepted.

Current managed models are:

| Model | Source label | Checkpoint step | Execution default |
| --- | --- | ---: | --- |
| LingBot VA fruit placement EEF | `step-1000` | 1000 | live, finite closed-loop rollout after task selection |
| OpenPI pi0.5 fruit placement EEF | `step-14999` | 14999 | live, finite closed-loop rollout after task selection |

Do not commit weights and do not add Git LFS. Do not delete artifacts or staging
directories without explicit review and authorization.
