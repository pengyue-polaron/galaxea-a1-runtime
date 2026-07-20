"""Strict scripted live-run plans for LingBot prompt repetitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from galaxea_a1_runtime.apps.lingbot.config import load_lingbot_config
from galaxea_a1_runtime.apps.lingbot.config_schema import LingBotConfig
from galaxea_a1_runtime.apps.reset.config import load_a1_home_pose
from galaxea_a1_runtime.configuration.base import (
    integer,
    load_toml,
    lower_identifier,
    repo_path,
    require_exact_keys,
    required_table,
    shell_assign,
    string,
)
from galaxea_a1_runtime.console import ArgumentParser
from galaxea_a1_runtime.configuration.paths import LINGBOT_BATCH_CONFIG


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BATCH_CONFIG = ROOT / LINGBOT_BATCH_CONFIG


@dataclass(frozen=True)
class LingBotBatchConfig:
    path: Path
    batch_id: str
    deployment_path: Path
    deployment: LingBotConfig
    reset_pose: Path
    retries_per_prompt: int
    task_ids: tuple[str, ...]

    @property
    def attempts_per_prompt(self) -> int:
        return self.retries_per_prompt + 1

    @property
    def total_attempts(self) -> int:
        return len(self.task_ids) * self.attempts_per_prompt


def load_lingbot_batch_config(
    path: Path,
    *,
    repo_root: Path | None = None,
    model_selector: str | None = None,
) -> LingBotBatchConfig:
    path, root, data = load_toml(path, repo_root=repo_root)
    require_exact_keys(
        data,
        required={"deployment", "reset", "batch"},
        label="LingBot batch config",
    )
    deployment_ref = required_table(data, "deployment")
    require_exact_keys(
        deployment_ref, required={"config"}, label="batch deployment reference"
    )
    deployment_path = repo_path(root, string(deployment_ref, "config"))
    deployment = load_lingbot_config(
        deployment_path,
        repo_root=root,
        model_selector=model_selector,
    )

    reset = required_table(data, "reset")
    require_exact_keys(reset, required={"pose"}, label="batch reset reference")
    reset_pose = repo_path(root, string(reset, "pose"))
    load_a1_home_pose(reset_pose, system=deployment.system, repo_root=root)

    batch = required_table(data, "batch")
    require_exact_keys(
        batch,
        required={"schema_version", "id", "retries_per_prompt", "task_ids"},
        label="batch plan",
    )
    if integer(batch, "schema_version") != 1:
        raise ValueError("batch.schema_version must be 1")
    batch_id = lower_identifier(string(batch, "id"), label="batch.id")
    retries = integer(batch, "retries_per_prompt")
    if retries < 0:
        raise ValueError("batch.retries_per_prompt must be non-negative")
    raw_task_ids = batch.get("task_ids")
    if not isinstance(raw_task_ids, list) or not raw_task_ids:
        raise ValueError("batch.task_ids must be a non-empty string list")
    if any(not isinstance(value, str) or not value for value in raw_task_ids):
        raise ValueError("batch.task_ids must contain non-empty strings")
    task_ids = tuple(raw_task_ids)
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("batch.task_ids must not contain duplicates")
    for task_id in task_ids:
        deployment.task_catalog.task(task_id)
    return LingBotBatchConfig(
        path=path,
        batch_id=batch_id,
        deployment_path=deployment_path,
        deployment=deployment,
        reset_pose=reset_pose,
        retries_per_prompt=retries,
        task_ids=task_ids,
    )


def bash_config(config: LingBotBatchConfig) -> str:
    values = (
        ("BATCH_ID", config.batch_id),
        ("BATCH_DEPLOYMENT_CONFIG", str(config.deployment_path)),
        ("BATCH_RESET_POSE", str(config.reset_pose)),
        ("BATCH_RETRIES_PER_PROMPT", str(config.retries_per_prompt)),
        ("BATCH_ATTEMPTS_PER_PROMPT", str(config.attempts_per_prompt)),
        ("BATCH_TOTAL_ATTEMPTS", str(config.total_attempts)),
        ("BATCH_TASK_IDS_CSV", ",".join(config.task_ids)),
    )
    return "\n".join(shell_assign(name, value) for name, value in values)


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--model")
    parser.add_argument("--shell", action="store_true")
    parser.add_argument("config", type=Path, nargs="?", default=DEFAULT_BATCH_CONFIG)
    args = parser.parse_args()
    config = load_lingbot_batch_config(
        args.config,
        repo_root=args.repo_root,
        model_selector=args.model,
    )
    print(bash_config(config) if args.shell else config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
