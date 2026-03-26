from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from config import settings

# Convert standard postgres:// URL to async postgresql+asyncpg:// URL
# asyncpg is the async driver that lets FastAPI talk to PostgreSQL without blocking
DATABASE_URL = settings.database_url.replace(
    "postgresql://", "postgresql+asyncpg://"
)

# Create the async engine — this is the core connection to PostgreSQL
# pool_pre_ping checks if connection is alive before using it
engine = create_async_engine(
    DATABASE_URL,
    echo=settings.environment == "development",  # logs SQL in dev mode
    pool_pre_ping=True,
    pool_size=10,          # max 10 simultaneous connections
    max_overflow=20,       # allow 20 extra connections under heavy load
)

# Session factory — creates new database sessions
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # keeps objects accessible after commit
)


# Base class — all database models inherit from this
class Base(DeclarativeBase):
    pass


# Creates all tables on startup if they don't exist
async def create_tables():
    async with engine.begin() as conn:
        from models.database import Session, Architecture, ChatMessage, CaseStudy, Benchmark
        await conn.run_sync(Base.metadata.create_all)


# Dependency injection for FastAPI routers
# Used as: async def my_route(db: AsyncSession = Depends(get_db))
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
