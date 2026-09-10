from __future__ import annotations

import json
import sys
from pathlib import Path

GT_DIR = Path(__file__).resolve().parent.parent.parent / "evaluation" / "ground_truth"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

STANDARDS = {"aws", "azure", "gcp", "c4", "uml", "informal"}
ARCH_TYPES = {"microservices", "monolith", "modular_monolith", "event_driven",
              "serverless", "layered", "client_server", "pipeline",
              "multi_region", "other"}
TYPES = {"service", "database", "gateway", "queue", "cache", "cdn",
         "load_balancer", "client", "storage", "monitoring", "notification",
         "other"}


def complexity_for(n: int) -> str:

    if n < 8:
        return "low"
    return "medium" if n <= 14 else "high"


def validate(path: Path, fix: bool) -> tuple[list[str], list[str], bool]:
    errors: list[str] = []
    warnings: list[str] = []
    changed = False

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"invalid JSON: {exc}"], [], False

    # ── Required top-level fields ─────────────────────────────────────────
    for field in ("diagram_id", "source_image", "diagram_standard",
                  "complexity", "components", "connections"):
        if field not in data:
            errors.append(f"missing required field '{field}'")
    if errors:
        return errors, warnings, False

    if data["diagram_id"] != path.stem:
        errors.append(f"diagram_id '{data['diagram_id']}' != filename '{path.stem}'")

    if data["diagram_standard"] not in STANDARDS:
        errors.append(f"diagram_standard '{data['diagram_standard']}' "
                      f"not in {sorted(STANDARDS)}")

    if data.get("arch_type") and data["arch_type"] not in ARCH_TYPES:
        warnings.append(f"arch_type '{data['arch_type']}' not in {sorted(ARCH_TYPES)}")

    # source_url is the evidence the annotation is faithful to a real diagram
    if not data.get("source_url"):
        errors.append("missing source_url — annotation has no provenance")

    # ── Image present and matching ────────────────────────────────────────
    image = next((path.with_suffix(e) for e in IMAGE_EXTS
                  if path.with_suffix(e).exists()), None)
    if image is None:
        errors.append("no image file alongside this annotation")
    elif data["source_image"] != image.name:
        warnings.append(f"source_image '{data['source_image']}' != actual '{image.name}'")
        if fix:
            data["source_image"] = image.name
            changed = True

    # ── Components ────────────────────────────────────────────────────────
    components = data["components"]
    if not components:
        errors.append("no components annotated")

    names: list[str] = []
    for i, component in enumerate(components):
        if not isinstance(component, dict) or "name" not in component:
            errors.append(f"component[{i}] has no 'name'")
            continue
        name = component["name"]
        if not str(name).strip():
            errors.append(f"component[{i}] has an empty name")
        if component.get("type") and component["type"] not in TYPES:
            warnings.append(f"component '{name}' type '{component['type']}' "
                            f"not in the schema enum")
        names.append(name)

    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        # Connections are matched by name, so duplicates are ambiguous targets.
        errors.append(f"duplicate component names: {sorted(duplicates)}")

    # ── Connections ───────────────────────────────────────────────────────
    known = set(names)
    for i, conn in enumerate(data["connections"]):
        if not isinstance(conn, dict) or "source" not in conn or "target" not in conn:
            errors.append(f"connection[{i}] missing source/target")
            continue
        for end in ("source", "target"):
            if conn[end] not in known:
                errors.append(
                    f"connection[{i}] {end} '{conn[end]}' is not a component name"
                )
        if conn["source"] == conn["target"]:
            warnings.append(f"connection[{i}] is a self-loop on '{conn['source']}'")

    pairs = [(c.get("source"), c.get("target")) for c in data["connections"]]
    dupe_pairs = {p for p in pairs if pairs.count(p) > 1}
    if dupe_pairs:
        warnings.append(f"duplicate connections: {sorted(dupe_pairs)}")

    # ── Derived fields ────────────────────────────────────────────────────
    expected = complexity_for(len(components))
    if data["complexity"] != expected:
        msg = (f"complexity '{data['complexity']}' disagrees with "
               f"{len(components)} components (expected '{expected}')")
        if fix:
            data["complexity"] = expected
            changed = True
            warnings.append(msg + " — fixed")
        else:
            errors.append(msg)

    if changed:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    return errors, warnings, changed


def main() -> int:
    fix = "--fix" in sys.argv
    files = sorted(GT_DIR.glob("*.json"))
    if not files:
        print(f"No annotations found in {GT_DIR}")
        return 1

    total_err = total_warn = fixed = 0
    by_standard: dict[str, int] = {}

    for path in files:
        errors, warnings, changed = validate(path, fix)
        data = {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
        std = data.get("diagram_standard", "?")
        by_standard[std] = by_standard.get(std, 0) + 1

        if errors or warnings:
            print(f"\n{path.name}")
            for e in errors:
                print(f"  ERROR  {e}")
            for w in warnings:
                print(f"  warn   {w}")
        total_err += len(errors)
        total_warn += len(warnings)
        fixed += int(changed)

    print("\n" + "=" * 62)
    print(f"{len(files)} annotation files: {total_err} errors, {total_warn} warnings"
          + (f", {fixed} fixed" % () if fix else ""))
    print("counts by notation: " + ", ".join(
        f"{k}={v}" for k, v in sorted(by_standard.items())))
    return 1 if total_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
