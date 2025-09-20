from __future__ import annotations

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get("DUMMY_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql+psycopg://user:pass@localhost:5432/cim_dummy"

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def run_sql(sql: str):
    with engine.connect() as conn:
        return conn.execute(text(sql))
