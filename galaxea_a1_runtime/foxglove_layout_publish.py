"""Publish one committed Foxglove layout as a canonical organization layout."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.console import ArgumentParser, success


FOXGLOVE_LAYOUTS_API = "https://api.foxglove.dev/v1/layouts"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class PublishedLayout:
    id: str
    name: str
    action: str


def select_layout_id(layouts: Any, *, name: str) -> str | None:
    if not isinstance(layouts, list):
        raise ValueError("Foxglove layout list response must be an array")
    matches: list[str] = []
    for index, item in enumerate(layouts):
        if not isinstance(item, dict):
            raise ValueError(f"Foxglove layout list item {index} must be an object")
        if item.get("name") != name:
            continue
        layout_id = item.get("id")
        if not isinstance(layout_id, str) or not layout_id:
            raise ValueError(f"Foxglove layout named {name!r} has no valid id")
        matches.append(layout_id)
    if len(matches) > 1:
        raise ValueError(
            f"Foxglove organization contains {len(matches)} layouts named {name!r}; "
            "refusing an ambiguous update"
        )
    return matches[0] if matches else None


def layout_payload(
    data: Any,
    *,
    name: str,
    folder: str,
    permission: str,
) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("committed Foxglove layout must contain one JSON object")
    if permission != "ORG_WRITE":
        raise ValueError("API-key-managed Foxglove layouts must use ORG_WRITE")
    if "/" in folder:
        raise ValueError("Foxglove layout folder cannot contain a forward slash")
    return {
        "name": name,
        "folderName": folder,
        "permission": permission,
        "data": data,
    }


class FoxgloveLayoutsClient:
    def __init__(self, *, api_key: str) -> None:
        if not api_key:
            raise ValueError("FOXGLOVE_API_KEY is required")
        self._api_key = api_key

    def request(
        self,
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        body = None
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": "galaxea-a1-runtime-layout-publisher/1",
        }
        if payload is not None:
            body = json.dumps(
                payload,
                allow_nan=False,
                separators=(",", ":"),
            ).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                response_body = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            error_body = exc.read(64 * 1024).decode(errors="replace")
            error_body = error_body.replace(self._api_key, "[REDACTED]")
            raise RuntimeError(
                f"Foxglove API {method} failed with HTTP {exc.code}: {error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Foxglove API {method} request failed: {exc}") from exc
        if len(response_body) > MAX_RESPONSE_BYTES:
            raise RuntimeError("Foxglove API response exceeded 8 MiB")
        try:
            return json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Foxglove API returned invalid JSON") from exc

    def upsert(
        self,
        data: Any,
        *,
        name: str,
        folder: str,
        permission: str,
    ) -> PublishedLayout:
        layouts = self.request("GET", f"{FOXGLOVE_LAYOUTS_API}?includeData=false")
        existing_id = select_layout_id(layouts, name=name)
        payload = layout_payload(
            data,
            name=name,
            folder=folder,
            permission=permission,
        )
        if existing_id is None:
            result = self.request("POST", FOXGLOVE_LAYOUTS_API, payload=payload)
            action = "created"
        else:
            encoded_id = urllib.parse.quote(existing_id, safe="")
            result = self.request(
                "PATCH",
                f"{FOXGLOVE_LAYOUTS_API}/{encoded_id}",
                payload=payload,
            )
            action = "updated"
        if not isinstance(result, dict):
            raise RuntimeError("Foxglove layout write response must be an object")
        result_id = result.get("id")
        result_name = result.get("name")
        if not isinstance(result_id, str) or result_name != name:
            raise RuntimeError("Foxglove layout write response identity mismatch")
        return PublishedLayout(id=result_id, name=result_name, action=action)


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--folder", default="Galaxea A1")
    parser.add_argument(
        "--permission",
        choices=("ORG_WRITE",),
        default="ORG_WRITE",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.layout.open(encoding="utf-8") as stream:
        data = json.load(stream)
    client = FoxgloveLayoutsClient(api_key=os.environ.get("FOXGLOVE_API_KEY", ""))
    result = client.upsert(
        data,
        name=args.name,
        folder=args.folder,
        permission=args.permission,
    )
    success(f"Foxglove layout {result.action}: {result.name} ({result.id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
