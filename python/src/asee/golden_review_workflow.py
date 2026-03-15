from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

DEFAULT_GOLDEN_REVIEW_ROOT = Path(
    "/home/yuiseki/Workspaces/private/datasets/faces/golden_review_workspaces"
)
DEFAULT_FIFTYONE_PYTHON = Path(
    "/home/yuiseki/Workspaces/repos/_vision/fiftyone/.venv/bin/python"
)
DEFAULT_LABEL_STUDIO_PYTHON = Path(
    "/home/yuiseki/Workspaces/repos/_vision/label-studio/.venv/bin/python"
)
DEFAULT_LABEL_CHOICES = (
    "owner_positive",
    "guest_negative",
    "non_face_negative",
    "uncertain",
)


@dataclass(frozen=True)
class GoldenReviewSample:
    sample_id: str
    source_image: str
    source_sidecar: str | None
    source_key: str
    asset_relpath: str
    existing_label: str
    camera_id: int | None
    score: float | None
    frame_counts: dict[str, int]
    metadata_json: str


@dataclass(frozen=True)
class GoldenReviewWorkspaceSummary:
    workspace_dir: Path
    sample_count: int
    manifest_path: Path
    label_studio_tasks_path: Path
    label_studio_config_path: Path


def build_source_key(source_root: Path) -> str:
    parts = [part for part in source_root.parts if part]
    tail = parts[-5:] if len(parts) >= 5 else parts
    text = "-".join(tail)
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in text)


def build_label_studio_config(
    choices: Iterable[str] = DEFAULT_LABEL_CHOICES,
) -> str:
    choice_lines = "\n".join(
        f'      <Choice value="{choice}"/>' for choice in choices
    )
    return f"""<View>
  <Header value="ASEE face crop golden review"/>
  <View style="display:flex; gap: 16px;">
    <View style="width: 72%;">
      <Image name="image" value="$image" zoom="true"/>
    </View>
    <View style="width: 28%;">
      <Header value="$sample_id"/>
      <Text name="source_key_meta" value="$source_key"/>
      <Text name="camera_meta" value="$camera_id"/>
      <Text name="existing_label_meta" value="$existing_label"/>
      <Text name="score_meta" value="$score"/>
      <Text name="frame_counts_meta" value="$frame_counts"/>
      <Choices name="gold_label" toName="image" choice="single-radio" required="true">
{choice_lines}
      </Choices>
      <TextArea name="review_notes" toName="image" rows="4" maxSubmissions="1"/>
    </View>
  </View>
</View>
"""


def _iter_image_paths(source_root: Path) -> list[Path]:
    return sorted(
        path
        for path in source_root.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def _load_sidecar(path: Path) -> dict[str, Any]:
    sidecar_path = path.with_suffix(".json")
    if not sidecar_path.exists():
        return {}
    return cast(dict[str, Any], json.loads(sidecar_path.read_text(encoding="utf-8")))


def _copy_or_symlink(src: Path, dst: Path, *, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def _build_metadata_json(sidecar: dict[str, Any]) -> str:
    payload = {
        "capturedAt": sidecar.get("capturedAt"),
        "cameraId": sidecar.get("cameraId"),
        "label": sidecar.get("label"),
        "score": sidecar.get("score"),
        "frameCounts": sidecar.get("frameCounts") or {},
        "faceBox": sidecar.get("faceBox"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _build_label_studio_task(sample: GoldenReviewSample) -> dict[str, Any]:
    relative_url = quote(sample.asset_relpath, safe="/")
    frame_counts = sample.frame_counts or {}
    frame_counts_text = (
        f"owner={frame_counts.get('ownerCount', 0)} "
        f"subject={frame_counts.get('subjectCount', 0)} "
        f"people={frame_counts.get('peopleCount', 0)}"
    )
    return {
        "id": sample.sample_id,
        "data": {
            "image": f"/data/local-files/?d={relative_url}",
            "sample_id": sample.sample_id,
            "source_key": sample.source_key,
            "camera_id": "" if sample.camera_id is None else str(sample.camera_id),
            "existing_label": sample.existing_label,
            "score": "" if sample.score is None else f"{sample.score:.2f}",
            "frame_counts": frame_counts_text,
        },
        "meta": {
            "source_image": sample.source_image,
            "source_sidecar": sample.source_sidecar,
            "metadata_json": sample.metadata_json,
        },
    }


def _build_fiftyone_launcher(
    *,
    workspace_dir: Path,
    manifest_path: Path,
    dataset_name: str,
    python_path: Path,
) -> str:
    fiftyone_repo_root = python_path.parent.parent.parent
    return f"""#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${{FIFTYONE_PYTHON:-{python_path}}}"
export PYTHONPATH="{fiftyone_repo_root}:${{PYTHONPATH:-}}"
exec "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
import fiftyone as fo

manifest_path = Path({str(manifest_path)!r})
dataset_name = {dataset_name!r}
if fo.dataset_exists(dataset_name):
    fo.delete_dataset(dataset_name)
dataset = fo.Dataset(dataset_name)
for line in manifest_path.read_text(encoding="utf-8").splitlines():
    if not line:
        continue
    row = json.loads(line)
    sample = fo.Sample(filepath=str(Path({str(workspace_dir)!r}) / row["asset_relpath"]))
    sample["sample_id"] = row["sample_id"]
    sample["source_key"] = row["source_key"]
    sample["existing_label"] = row["existing_label"]
    sample["camera_id"] = row["camera_id"]
    sample["score"] = row["score"]
    sample["frame_counts"] = row["frame_counts"]
    sample["source_image"] = row["source_image"]
    sample["source_sidecar"] = row["source_sidecar"]
    tags = [row["source_key"]]
    if row["existing_label"]:
        tags.append("existing:" + str(row["existing_label"]).lower())
    if row["camera_id"] is not None:
        tags.append("camera:" + str(row["camera_id"]))
    sample.tags = tags
    dataset.add_sample(sample)
session = fo.launch_app(dataset, auto=False)
print(f"FiftyOne URL: {{session.url}}", flush=True)
session.wait()
PY
"""


def _build_label_studio_launcher(*, workspace_dir: Path, python_path: Path) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${{LABEL_STUDIO_PYTHON:-{python_path}}}"
export LABEL_STUDIO_BASE_DATA_DIR="{workspace_dir / 'label_studio_data'}"
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="{workspace_dir}"
exec "$PYTHON_BIN" -c "from label_studio.server import main; main()"
"""


def _write_workspace_readme(
    *,
    workspace_dir: Path,
    summary: GoldenReviewWorkspaceSummary,
) -> None:
    readme = f"""# ASEE Golden Review Workspace

Prepared workspace: `{workspace_dir}`

## Contents

- `assets/`: review images and sidecar JSON files
- `manifest.jsonl`: unified metadata manifest
- `label_studio/tasks.json`: importable Label Studio tasks
- `label_studio/config.xml`: 4-way golden label config
- `launch_fiftyone.sh`: launch FiftyOne review app
- `launch_label_studio.sh`: launch Label Studio with local-files serving enabled

## Suggested workflow

1. Launch FiftyOne first and inspect clusters / failure modes:
   `./launch_fiftyone.sh`
   The launcher prints a local URL instead of stealing browser focus.
2. Launch Label Studio in another terminal:
   `./launch_label_studio.sh`
3. In Label Studio, create a project and import:
   - config: `label_studio/config.xml`
   - tasks: `label_studio/tasks.json`

## Workspace summary

- samples: {summary.sample_count}
- manifest: `{summary.manifest_path}`
- tasks: `{summary.label_studio_tasks_path}`
"""
    (workspace_dir / "README.md").write_text(readme, encoding="utf-8")


def prepare_golden_review_workspace(
    *,
    sources: Iterable[Path],
    workspace_dir: Path,
    copy_files: bool = False,
    fiftyone_python: Path = DEFAULT_FIFTYONE_PYTHON,
    label_studio_python: Path = DEFAULT_LABEL_STUDIO_PYTHON,
) -> GoldenReviewWorkspaceSummary:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = workspace_dir / "assets"
    label_studio_dir = workspace_dir / "label_studio"
    assets_dir.mkdir(parents=True, exist_ok=True)
    label_studio_dir.mkdir(parents=True, exist_ok=True)

    samples: list[GoldenReviewSample] = []
    for source_root in [Path(source) for source in sources]:
        source_key = build_source_key(source_root)
        for image_index, image_path in enumerate(_iter_image_paths(source_root), start=1):
            sidecar = _load_sidecar(image_path)
            rel_under_source = image_path.relative_to(source_root)
            asset_relpath = str((Path("assets") / source_key / rel_under_source).as_posix())
            asset_image_path = workspace_dir / asset_relpath
            _copy_or_symlink(image_path, asset_image_path, copy_files=copy_files)

            sidecar_path = image_path.with_suffix(".json")
            asset_sidecar_path = asset_image_path.with_suffix(".json")
            if sidecar_path.exists():
                _copy_or_symlink(sidecar_path, asset_sidecar_path, copy_files=copy_files)

            sample_id = f"{source_key}-{image_index:04d}"
            samples.append(
                GoldenReviewSample(
                    sample_id=sample_id,
                    source_image=str(image_path),
                    source_sidecar=str(sidecar_path) if sidecar_path.exists() else None,
                    source_key=source_key,
                    asset_relpath=asset_relpath,
                    existing_label=str(sidecar.get("label") or ""),
                    camera_id=(
                        int(sidecar["cameraId"])
                        if isinstance(sidecar.get("cameraId"), int)
                        else None
                    ),
                    score=(
                        float(sidecar["score"])
                        if isinstance(sidecar.get("score"), (int, float))
                        else None
                    ),
                    frame_counts=dict(sidecar.get("frameCounts") or {}),
                    metadata_json=_build_metadata_json(sidecar),
                )
            )

    manifest_path = workspace_dir / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(asdict(sample), ensure_ascii=False) for sample in samples) + "\n",
        encoding="utf-8",
    )

    tasks_path = label_studio_dir / "tasks.json"
    tasks_path.write_text(
        json.dumps(
            [_build_label_studio_task(sample) for sample in samples],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    config_path = label_studio_dir / "config.xml"
    config_path.write_text(build_label_studio_config(), encoding="utf-8")

    dataset_name = f"asee-golden-review-{workspace_dir.name}"
    fiftyone_launcher_path = workspace_dir / "launch_fiftyone.sh"
    fiftyone_launcher_path.write_text(
        _build_fiftyone_launcher(
            workspace_dir=workspace_dir,
            manifest_path=manifest_path,
            dataset_name=dataset_name,
            python_path=fiftyone_python,
        ),
        encoding="utf-8",
    )
    fiftyone_launcher_path.chmod(0o755)

    label_studio_launcher_path = workspace_dir / "launch_label_studio.sh"
    label_studio_launcher_path.write_text(
        _build_label_studio_launcher(
            workspace_dir=workspace_dir,
            python_path=label_studio_python,
        ),
        encoding="utf-8",
    )
    label_studio_launcher_path.chmod(0o755)

    summary = GoldenReviewWorkspaceSummary(
        workspace_dir=workspace_dir,
        sample_count=len(samples),
        manifest_path=manifest_path,
        label_studio_tasks_path=tasks_path,
        label_studio_config_path=config_path,
    )
    _write_workspace_readme(workspace_dir=workspace_dir, summary=summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a golden review workspace")
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=DEFAULT_GOLDEN_REVIEW_ROOT,
        help="Root directory that will contain generated review workspaces.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Workspace name under the workspace root.",
    )
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        required=True,
        help="Source directory containing face crop images and sidecar JSON files.",
    )
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy source files instead of creating symlinks.",
    )
    parser.add_argument(
        "--fiftyone-python",
        type=Path,
        default=DEFAULT_FIFTYONE_PYTHON,
        help="Python interpreter used by launch_fiftyone.sh.",
    )
    parser.add_argument(
        "--label-studio-python",
        type=Path,
        default=DEFAULT_LABEL_STUDIO_PYTHON,
        help="Python interpreter used by launch_label_studio.sh.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    workspace_dir = args.workspace_root / args.name
    summary = prepare_golden_review_workspace(
        sources=args.source,
        workspace_dir=workspace_dir,
        copy_files=bool(args.copy_files),
        fiftyone_python=args.fiftyone_python,
        label_studio_python=args.label_studio_python,
    )
    print(f"workspace={summary.workspace_dir}")
    print(f"samples={summary.sample_count}")
    print(f"manifest={summary.manifest_path}")
    print(f"label_studio_tasks={summary.label_studio_tasks_path}")
    print(f"label_studio_config={summary.label_studio_config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
