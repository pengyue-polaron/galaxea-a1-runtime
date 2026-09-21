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

Task registries use a static `catalog.json` identity plus one strict
`prompts/<task-id>.json` file per approved prompt. The Web panel may register a
new create-only record, but runtime selection still accepts only a task that the
fully loaded registry already contains.

LingBot deployments contain a registered default, while their command-line
entrypoints accept `--model`. The selector resolves only an exact registered
model id, a unique descriptor filename, or a pinned id/revision; it never
accepts an arbitrary weight directory. For example:

```bash
just lingbot --model mango_placement_eef
just lingbot-verify --model mango_placement_eef
just lingbot-batch --model mango_placement_eef configs/runs/lingbot/mango_placement.toml
```

## Integrity and publication

Every managed model pins an immutable Hugging Face commit. Its tracked manifest
lists the exact non-cache file set, byte size, and SHA-256 of every file.
Download occurs in a hidden sibling staging directory. Identical files from an
already present immutable artifact are reused by verified content hash on the
same filesystem; only missing content is requested from the Hub. Only after
full validation does an atomic rename expose the final revision directory. A
crash leftover intentionally blocks reuse until it is inspected.

Fetch or verify one descriptor directly:

```bash
just model-fetch configs/models/pi05/fruit_placement_eef.toml
just model-verify configs/models/pi05/fruit_placement_eef.toml
```

Validate every configured model:

```bash
just models
```

## Release families

One pinned Hub revision often carries many checkpoints of the same
architecture. A tracked release plan registers the whole family instead of
hand-writing four files per model:

```toml
[release]
schema_version = 1
id = "distill-wam-a1-20260922"

[source]
provider = "huggingface"
repo_id = "SeanZheng/Distill-WAM"
revision = "0507667b5d4428c5c5478acefa6106389a61709b"
revision_label = "a1-kitchen-and-flashwam"

[ports]
server = 1130
master = 29600

[[model]]
key = "kitchen_pour_water_student"
id = "diffusion2one/a1_kitchen_pour_water_eef"
directory = "galaxea-a1-kitchen-pour-water"
checkpoint_step = 1000
kind = "student"
catalog = "configs/tasks/kitchen_pour_water/catalog.json"
```

`kind` selects the reviewed runtime class (`student` → `diffusion2one` backend,
`teacher` → `diffusion2one_teacher`) with its deployment and contract
templates. Ports take the table value plus the model order and must stay free
of other tracked deployments.

```bash
just model-register-release configs/releases/distill_wam_a1_20260922.toml \
    --from-local models/Distill-WAM
just model-register-release configs/releases/distill_wam_a1_20260922.toml \
    --only kitchen_pour_water_student
```

The registrar requires every release folder to carry exactly the reviewed file
set, proves each file against the pinned revision's Hub tree (LFS SHA-256, git
blob hash for small metadata), hard-links verified local content into hidden
staging, hashes it once, publishes the artifact by atomic rename, and generates
the descriptor, manifest, contract, and deployment as create-only files. An
edited generated file is a conflict, never an overwrite. Receipts are written
under `outputs/model_registration/<release-id>_<timestamp>/`.

The contract owns `components.model_subdirectory`; the loader resolves every
required manifest path through it instead of pinning one folder name. Each
release model references one task catalog, and a catalog holds one or more
approved prompts, so a single-prompt model and a multi-prompt model use the
same mechanism.

## Selecting a registered model

```bash
just inference --list                            # deployments, models, prompts, artifact state
just inference                                   # pick a target, then a prompt when needed
just inference kitchen_pour_water_student        # one-prompt catalog starts directly
just inference kitchen_no_memory_teacher --task pot_on_stove_turn_on_switch
just inference kitchen_pour_water_flashwam_step250 --action smoke
```

Selection accepts a deployment id or path, a model id, or a descriptor name.
Only a ready artifact, a registered model, and a tracked prompt can start.
`--action server` and `--action smoke` are hardware-free; `run` is the guarded
live rollout and may move the A1.

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
| LingBot VA mango-to-plate EEF | `step-100` | 100 | selectable as `mango_plate_eef` |
| LingBot VA mango placement EEF | `step-200` | 200 | selectable as `mango_placement_eef`; tracked full-catalog batch plan available |
| LingBot VA plug insertion EEF | `step-500` | 500 | dedicated first-socket deployment and three-attempt batch plan available |
| OpenPI pi0.5 fruit placement EEF | `step-14999` | 14999 | live, finite closed-loop rollout after task selection |
| Distill-WAM A1 fruit/blocks student | `galaxea-a1-ema-step-1000` | 1000 | single-step, live guarded rollout |
| Distill-WAM A1 fruit/blocks teacher | `galaxea-a1-posttrain-step-1000` | 1000 | `just diffusion2one-teacher` starts model server only; `run` and `batch` execute guarded motion |

The teacher is registered as `diffusion2one/a1_fruit_blocks_teacher_eef` at
`d1fa5b95b41fa96408be5c37cfae2ce900d1c615`, with a dedicated multi-step backend
and `galaxea-a1-teacher/` artifact prefix. Its imported Torch files are checked
against the release manifest before being published under `models/artifacts/`.
The existing student backend cannot select this teacher. See the
[teacher runbook](../docs/RUNBOOK.md#a1-teacher) for hardware-free verification
and inference commands.

Do not commit weights and do not add Git LFS. Do not delete artifacts or staging
directories without explicit review and authorization.

The transferred TFP checkpoint identity and metadata hashes live in
`configs/models/tfp/press_button_joint.checkpoint.toml`; its deployment references that
owner. TFP execution chunk length and horizon are read from the checkpoint by
the pinned runner, and diagnostics report its actual `n_action_steps`.
