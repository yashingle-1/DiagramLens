"""
Quick standalone test for the hybrid SAM+CLIP+TrOCR pipeline.

Usage:
    venv/Scripts/python.exe test_hybrid.py [image_path]

Defaults to a ground truth diagram so output quality can be eyeballed
against the annotation.
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from services.hybrid_pipeline import run_hybrid_pipeline

DEFAULT_IMAGE = Path(__file__).parent.parent / "evaluation" / "ground_truth" / "aws_three_tier_web.png"


async def main() -> None:
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    print(f"Image: {image_path}")
    image_bytes = image_path.read_bytes()

    start = time.time()
    result = await run_hybrid_pipeline(image_bytes, session_id="test-hybrid")
    wall = time.time() - start

    print(f"\npipeline={result.pipeline}  standard={result.diagram_standard}  "
          f"complexity={result.complexity}  arch_type={result.arch_type}")
    print(f"response_time_ms={result.response_time_ms}  (wall {wall:.1f}s)\n")

    print(f"Components ({len(result.components)}):")
    for c in result.components:
        print(f"  [{c.id}] {c.name!r}  type={c.type}  conf={c.confidence}")

    print(f"\nConnections ({len(result.connections)}):")
    for conn in result.connections:
        print(f"  {conn.source} -> {conn.target}")


if __name__ == "__main__":
    asyncio.run(main())
