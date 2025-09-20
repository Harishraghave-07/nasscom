from __future__ import annotations

from sqlalchemy import Column, Integer, String, Date, Numeric, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class DummyName(Base):
    __tablename__ = "dummy_names"
    id = Column(Integer, primary_key=True)
    first = Column(String(128), nullable=False)
    last = Column(String(128), nullable=False)
    full = Column(String(256), nullable=False)


class DummyAddress(Base):
    __tablename__ = "dummy_addresses"
    id = Column(Integer, primary_key=True)
    street = Column(String(256), nullable=False)
    city = Column(String(128), nullable=False)
    state = Column(String(64), nullable=False)
    zip = Column(String(16), nullable=False)


class DummyIdentifier(Base):
    __tablename__ = "dummy_identifiers"
    id = Column(Integer, primary_key=True)
    ssn = Column(String(32))
    phone = Column(String(64))
    email = Column(String(256))
    medical_id = Column(String(128))


class DummyDateFinancial(Base):
    __tablename__ = "dummy_date_financial"
    id = Column(Integer, primary_key=True)
    event_date = Column(Date)
    amount = Column(Numeric(12, 2))
    currency = Column(String(8))
    note = Column(Text)
