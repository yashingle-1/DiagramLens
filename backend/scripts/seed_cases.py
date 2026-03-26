"""
Seed script — loads industry case studies into PostgreSQL.

Run once after setting up the database:
    python scripts/seed_cases.py

Safe to run multiple times — uses upsert logic.
"""
import asyncio
import json
import os
import sys

# Add parent directory to path so we can import from backend
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.ext.asyncio import AsyncSession
from db.connection import AsyncSessionLocal, create_tables
from models.database import CaseStudy


CASES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cases")


async def seed():
    print("🌱 Starting case study seeding...")

    # Create tables if they don't exist
    await create_tables()

    # Load all JSON files from data/cases/
    json_files = [f for f in os.listdir(CASES_DIR) if f.endswith(".json")]

    if not json_files:
        print("❌ No JSON files found in data/cases/")
        return

    async with AsyncSessionLocal() as db:
        for filename in json_files:
            filepath = os.path.join(CASES_DIR, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            case_id = data["id"]

            # Check if case study already exists
            from sqlalchemy import select
            result = await db.execute(
                select(CaseStudy).where(CaseStudy.id == case_id)
            )
            existing = result.scalar_one_or_none()

            if existing:
                # Update existing
                existing.title = data["title"]
                existing.company = data["company"]
                existing.description = data.get("description")
                existing.difficulty = data.get("difficulty", "intermediate")
                existing.tags = data.get("tags", [])
                existing.architecture_json = data["architecture"]
                existing.hld_content = data.get("hld_content")
                existing.lld_content = data.get("lld_content")
                existing.flashcards = data.get("flashcards", [])
                print(f"  ✅ Updated: {data['title']}")
            else:
                # Insert new
                case = CaseStudy(
                    id=case_id,
                    title=data["title"],
                    company=data["company"],
                    description=data.get("description"),
                    difficulty=data.get("difficulty", "intermediate"),
                    tags=data.get("tags", []),
                    architecture_json=data["architecture"],
                    hld_content=data.get("hld_content"),
                    lld_content=data.get("lld_content"),
                    flashcards=data.get("flashcards", []),
                    is_active=True,
                )
                db.add(case)
                print(f"  ✅ Inserted: {data['title']}")

        await db.commit()
        print(f"\n🎉 Seeded {len(json_files)} case studies successfully!")


if __name__ == "__main__":
    asyncio.run(seed())
