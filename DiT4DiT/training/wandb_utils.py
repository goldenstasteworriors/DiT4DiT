"""Utilities for assigning W&B projects from dataset task prompts."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any


def _canonical_prompt(prompt: Any) -> str:
    if prompt is None:
        return ""
    return " ".join(str(prompt).split()).strip()


def _project_slug(prompt: str, max_length: int = 120) -> str:
    canonical = _canonical_prompt(prompt).casefold()
    characters: list[str] = []
    previous_was_separator = False
    for character in canonical:
        if character.isalnum():
            characters.append(character)
            previous_was_separator = False
        elif not previous_was_separator:
            characters.append("_")
            previous_was_separator = True

    slug = "".join(characters).strip("_")
    if not slug:
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
        slug = f"dit4dit_task_{digest}"
    if len(slug) > max_length:
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
        slug = f"{slug[: max_length - len(digest) - 1].rstrip('_')}_{digest}"
    return slug


def _prompts_from_tasks(tasks: Any) -> list[str]:
    if tasks is None:
        return []

    if hasattr(tasks, "columns") and "task" in tasks.columns:
        values = tasks["task"].tolist()
    elif isinstance(tasks, dict):
        values = tasks.get("task", [])
        if isinstance(values, str):
            values = [values]
    elif isinstance(tasks, Iterable) and not isinstance(tasks, (str, bytes)):
        values = [item.get("task") for item in tasks if isinstance(item, dict)]
    else:
        values = []

    return [prompt for value in values if (prompt := _canonical_prompt(value))]


def collect_dataset_prompts(dataset: Any) -> list[str]:
    """Collect unique task prompts from a dataset or nested mixture dataset."""
    prompts: set[str] = set()
    pending = [dataset]
    visited: set[int] = set()

    while pending:
        current = pending.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))

        for prompt in _prompts_from_tasks(getattr(current, "tasks", None)):
            prompts.add(prompt.casefold())

        nested_datasets = getattr(current, "datasets", None)
        if isinstance(nested_datasets, Iterable) and not isinstance(nested_datasets, (str, bytes)):
            pending.extend(nested_datasets)

        nested_dataset = getattr(current, "dataset", None)
        if nested_dataset is not None and nested_dataset is not current:
            pending.append(nested_dataset)

    return sorted(prompts)


def resolve_wandb_project(
    dataset: Any,
    fallback_project: str,
    group_by_prompt: bool = True,
) -> tuple[str, list[str]]:
    """Resolve a stable W&B project name for the dataset's prompt set."""
    prompts = collect_dataset_prompts(dataset)
    if not group_by_prompt or not prompts:
        return fallback_project, prompts
    if len(prompts) == 1:
        return _project_slug(prompts[0]), prompts

    canonical_prompt_set = "\n".join(prompt.casefold() for prompt in prompts)
    digest = hashlib.sha256(canonical_prompt_set.encode("utf-8")).hexdigest()[:10]
    return f"dit4dit_{len(prompts)}_prompts_{digest}", prompts
