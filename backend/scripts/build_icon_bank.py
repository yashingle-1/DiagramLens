"""
Builds the icon embedding bank used by proposer P1 (services/icon_bank.py).

Run ONCE offline. At inference the hybrid arm only does a forward pass on the
crop plus a cosine lookup — the icons themselves are never re-embedded.

Layout expected:

    backend/data/icons/
        aws/    Arch_AWS-Lambda_48.png ...
        azure/  ...
        gcp/    ...
        mapping.json          # optional: filename stem -> {name, type}

Icon sources (free, official, redistributable for academic use record the
licence in the dissertation):
    AWS    https://aws.amazon.com/architecture/icons/
    Azure  https://learn.microsoft.com/en-us/azure/architecture/icons/
    GCP    https://cloud.google.com/icons

Without mapping.json the product name is derived from the filename, which is
usually close ("Arch_AWS-Lambda_48.png" -> "AWS Lambda") but should be
hand-corrected once and committed.

    python backend/scripts/build_icon_bank.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.common import classify_type          # noqa: E402
from services.icon_bank import BANK_PATH, MAPPING_PATH, embed_images  # noqa: E402

ICON_DIR = BANK_PATH.parent
PROVIDERS = ("aws", "azure", "gcp")
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
BATCH = 64

# Vendor filename decorations: size suffixes, resolution tags, arch prefixes.
_STRIP = re.compile(
    r"^(arch|res|icon|logo)[-_]|[-_](\d{2,4}|light|dark|squid|service|resource)$",
    re.IGNORECASE,
)


def _pretty_name(stem: str) -> str:
    name = stem
    for _ in range(3):
        name = _STRIP.sub("", name)
    name = re.sub(r"[-_]+", " ", name)
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)   # AmazonECS -> Amazon ECS
    return re.sub(r"\s+", " ", name).strip()


def main() -> int:
    if not ICON_DIR.is_dir():
        print(f"No icon directory at {ICON_DIR}")
        print("Create it and drop the vendor icon packs into aws/ azure/ gcp/ first.")
        return 1

    overrides: dict[str, dict] = {}
    if MAPPING_PATH.is_file():
        overrides = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))

    files: list[tuple[Path, str]] = []
    for provider in PROVIDERS:
        folder = ICON_DIR / provider
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() in EXTENSIONS:
                files.append((path, provider))

    if not files:
        print(f"No icon images found under {ICON_DIR}/{{{','.join(PROVIDERS)}}}")
        return 1

    print(f"Embedding {len(files)} icons...")
    vectors: list[np.ndarray] = []
    meta: list[str] = []

    for start in range(0, len(files), BATCH):
        chunk = files[start:start + BATCH]
        images = []
        for path, _ in chunk:
            # Flatten alpha onto white: vendor icons ship transparent, and a
            # black RGB conversion would not resemble the rendered diagram.
            img = Image.open(path).convert("RGBA")
            flat = Image.new("RGBA", img.size, (255, 255, 255, 255))
            flat.alpha_composite(img)
            images.append(flat.convert("RGB"))

        vectors.append(embed_images(images))
        for path, provider in chunk:
            override = overrides.get(path.stem, {})
            name = override.get("name") or _pretty_name(path.stem)
            meta.append(json.dumps({
                "name":     name,
                "type":     override.get("type") or classify_type(name),
                "provider": provider,
            }))
        print(f"  {min(start + BATCH, len(files))}/{len(files)}")

    BANK_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        BANK_PATH,
        vectors=np.vstack(vectors).astype(np.float32),
        meta=np.array(meta, dtype=np.str_),   # plain unicode: loads without pickle
    )
    print(f"Wrote {BANK_PATH}  ({len(meta)} icons)")

    if not MAPPING_PATH.is_file():
        seed = {
            Path(p).stem: {"name": json.loads(m)["name"], "type": json.loads(m)["type"]}
            for (p, _), m in zip(files, meta)
        }
        MAPPING_PATH.write_text(json.dumps(seed, indent=2), encoding="utf-8")
        print(f"Seeded {MAPPING_PATH} — correct the names by hand, then re-run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
