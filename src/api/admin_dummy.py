from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from src.services.dummy_data.service import (
    create_tables,
    seed_defaults,
    generate_addresses,
    generate_identifiers,
    generate_dates_financial,
    smart_replace_relationships,
)
from src.api.auth import require_role

router = APIRouter(prefix="/api/v1/admin/dummy-data", tags=["admin-dummy-data"])


class GenerateRequest(BaseModel):
    category: str
    count: int = 100


@router.get("/")
async def list_dummy_data(user=Depends(require_role("admin"))):
    # simple summary
    return {"status": "ok", "categories": ["names", "addresses", "identifiers", "dates_financial"]}


@router.post("/generate")
async def generate(req: GenerateRequest, user=Depends(require_role("admin"))):
    cat = req.category.lower()
    c = int(req.count or 100)
    if cat == "names":
        seed_defaults()
        return {"status": "ok", "generated": c}
    if cat == "addresses":
        generate_addresses(c)
        return {"status": "ok", "generated": c}
    if cat == "identifiers":
        generate_identifiers(c)
        return {"status": "ok", "generated": c}
    if cat == "dates_financial":
        generate_dates_financial(c)
        return {"status": "ok", "generated": c}
    raise HTTPException(status_code=400, detail="Unknown category")


@router.put("/{category}")
async def update_category(category: str, payload: Dict[str, Any], user=Depends(require_role("admin"))):
    # minimal update: allow admin to insert a single record via payload
    if category == "names":
        from src.services.dummy_data.db import SessionLocal
        from src.services.dummy_data.models import DummyName
        with SessionLocal() as s:
            n = DummyName(first=payload.get("first"), last=payload.get("last"), full=payload.get("full"))
            s.add(n)
            s.commit()
            return {"status": "ok", "id": n.id}
    # more categories can be handled similarly
    raise HTTPException(status_code=400, detail="Unsupported category")
