"""Standalone test for the classical pipeline. No Gemini, no DB."""
import asyncio
import sys
import glob
from services.classical_pipeline import run_classical_pipeline


async def run(path: str):
    with open(path, "rb") as f:
        data = f.read()
    res = await run_classical_pipeline(data, session_id="test")
    print(f"\n=== {path} ===")
    print(f"standard={res.diagram_standard}  complexity={res.complexity}  "
          f"arch={res.arch_type}  time={res.response_time_ms}ms")
    print(f"components ({len(res.components)}):")
    for c in res.components:
        print(f"  [{c.type:13}] {c.name}")
    print(f"connections ({len(res.connections)}):")
    by_id = {c.id: c.name for c in res.components}
    for e in res.connections:
        print(f"  {by_id.get(e.source, e.source)} -> {by_id.get(e.target, e.target)}")


async def main():
    args = sys.argv[1:]
    if not args:
        args = sorted(glob.glob("uploads/*.png")) + sorted(glob.glob("test_image*.png"))
    for p in args:
        await run(p)


if __name__ == "__main__":
    asyncio.run(main())
