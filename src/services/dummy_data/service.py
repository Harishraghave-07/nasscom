from __future__ import annotations

from datetime import date, timedelta
import random
from typing import List

from .db import run_sql, SessionLocal, engine
from .models import DummyName, DummyAddress, DummyIdentifier, DummyDateFinancial, Base
from sqlalchemy import text


def create_tables():
    Base.metadata.create_all(bind=engine)


def seed_defaults():
    # Create some basic names/addresses via SQL generate_series (if postgres available)
    sql_names = """
    INSERT INTO dummy_names (first, last, full)
    SELECT
      md5(random()::text)::text || '_' || gs::text as first,
      md5(random()::text)::text || '_' || (gs+100)::text as last,
      md5(random()::text)::text || '_' || (gs+200)::text as full
    FROM generate_series(1, 100) as gs;
    """
    try:
        run_sql(sql_names)
    except Exception:
        # fallback to ORM inserts
        from sqlalchemy.orm import Session

        with Session(engine) as s:
            for i in range(100):
                n = DummyName(first=f"First{i}", last=f"Last{i}", full=f"First{i} Last{i}")
                s.add(n)
            s.commit()


def generate_addresses(n: int = 100):
    # Try using SQL for faster bulk insertion
    try:
        sql = f"""
        INSERT INTO dummy_addresses (street, city, state, zip)
        SELECT
          (gs || ' ' || 'Main St')::text as street,
          CASE ((gs % 10)+1)
            WHEN 1 THEN 'Springfield' WHEN 2 THEN 'Riverton' WHEN 3 THEN 'Fairview'
            WHEN 4 THEN 'Greenville' WHEN 5 THEN 'Franklin' WHEN 6 THEN 'Bristol'
            WHEN 7 THEN 'Clinton' WHEN 8 THEN 'Madison' WHEN 9 THEN 'Georgetown' ELSE 'Oakland' END as city,
          'CA' as state,
          lpad((10000 + (gs % 90000))::text, 5, '0') as zip
        FROM generate_series(1, {n}) as gs;
        """
        run_sql(sql)
        return True
    except Exception:
        # fallback to ORM
        from sqlalchemy.orm import Session
        with Session(engine) as s:
            for i in range(n):
                a = DummyAddress(street=f"{i} Main St", city=random.choice(["Springfield", "Riverton", "Fairview"]), state="CA", zip=f"{90000+i%1000:05d}")
                s.add(a)
            s.commit()
        return True


def generate_identifiers(n: int = 100):
    # generate plausible SSNs and phones via SQL
    try:
        sql = f"""
        INSERT INTO dummy_identifiers (ssn, phone, email, medical_id)
        SELECT
          lpad((100000000 + (gs))::text,9,'0') as ssn,
          '+1'||lpad((2000000000 + gs)::text,10,'0') as phone,
          'user'||gs||'@example.com' as email,
          'MID'||(100000 + gs)::text as medical_id
        FROM generate_series(1, {n}) as gs;
        """
        run_sql(sql)
        return True
    except Exception:
        from sqlalchemy.orm import Session
        with Session(engine) as s:
            for i in range(n):
                ident = DummyIdentifier(ssn=f"{100000000+i}", phone=f"+1{2000000000+i}", email=f"user{i}@example.com", medical_id=f"MID{100000+i}")
                s.add(ident)
            s.commit()
        return True


def generate_dates_financial(n: int = 100):
    try:
        sql = f"""
        INSERT INTO dummy_date_financial (event_date, amount, currency, note)
        SELECT
          (CURRENT_DATE - (gs % 365))::date as event_date,
          (random()*10000)::numeric(12,2) as amount,
          'USD' as currency,
          'auto-generated' as note
        FROM generate_series(1, {n}) as gs;
        """
        run_sql(sql)
        return True
    except Exception:
        from sqlalchemy.orm import Session
        from datetime import date, timedelta
        with Session(engine) as s:
            for i in range(n):
                d = DummyDateFinancial(event_date=date.today() - timedelta(days=random.randint(0, 365)), amount=round(random.random() * 10000, 2), currency="USD", note="auto-generated")
                s.add(d)
            s.commit()
        return True


def smart_replace_relationships():
    # Placeholder: maintain relationships, e.g., map identifiers to names one-to-one
    # Example: ensure number of identifiers matches names and create mapping table if needed
    return True
