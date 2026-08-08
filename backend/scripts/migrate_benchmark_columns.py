"""
Adds the Part A benchmark columns to an existing database.

SQLAlchemy's create_all() only creates missing TABLES, never missing columns,
so an existing benchmarks table needs this. Idempotent — safe to re-run.

    python backend/scripts/migrate_benchmark_columns.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import text
from db.connection import engine

COLUMNS = [
    "connection_undirected_precision DOUBLE PRECISION",
    "connection_undirected_recall    DOUBLE PRECISION",
    "connection_undirected_f1        DOUBLE PRECISION",
    "component_f1_raw                DOUBLE PRECISION",
]


async def main() -> None:
    async with engine.begin() as conn:
        for col in COLUMNS:
            await conn.execute(
                text(f"ALTER TABLE benchmarks ADD COLUMN IF NOT EXISTS {col}")
            )
            print(f"ok  {col.split()[0]}")
    await engine.dispose()
    print("done")


if __name__ == "__main__":
    asyncio.run(main())
