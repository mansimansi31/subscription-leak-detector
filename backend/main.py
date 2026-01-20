from datetime import datetime, timedelta
from typing import Optional, List
import re

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from sqlalchemy import create_engine, text

app = FastAPI(title="Subscription Leak Detector API")

# local database
engine = create_engine("sqlite:///./app.db", future=True)

def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            merchant TEXT NOT NULL,
            amount REAL NOT NULL,
            description TEXT
        );
        """))
init_db()

@app.get("/health")
def health():
    return {"status": "ok"}

def normalize_merchant(name: str) -> str:
    n = name.lower().strip()
    n = re.sub(r"[^a-z0-9\s]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n

def detect_cadence(gaps: List[int]) -> Optional[str]:
    if len(gaps) < 2:
        return None
    median = sorted(gaps)[len(gaps)//2]
    if 25 <= median <= 35:
        return "monthly"
    if 350 <= median <= 380:
        return "yearly"
    if 5 <= median <= 9:
        return "weekly"
    return None

def predict_next(last_date: datetime, cadence: str) -> datetime:
    if cadence == "monthly":
        return last_date + timedelta(days=30)
    if cadence == "yearly":
        return last_date + timedelta(days=365)
    if cadence == "weekly":
        return last_date + timedelta(days=7)
    return last_date

@app.post("/transactions/upload")
async def upload_transactions(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload must be a .csv file")

    df = pd.read_csv(file.file)
    df.columns = [c.lower().strip() for c in df.columns]

    if not {"date", "merchant", "amount"}.issubset(df.columns):
        raise HTTPException(status_code=400, detail="CSV must include columns: date, merchant, amount")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["merchant"] = df["merchant"].astype(str).map(normalize_merchant)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    if "description" not in df.columns:
        df["description"] = None

    df = df.dropna(subset=["date", "merchant", "amount"])

    inserted = 0
    with engine.begin() as conn:
        for _, r in df.iterrows():
            conn.execute(
                text("""
                INSERT INTO transactions(date, merchant, amount, description)
                VALUES (:date, :merchant, :amount, :desc)
                """),
                {
                    "date": r["date"].strftime("%Y-%m-%d"),
                    "merchant": r["merchant"],
                    "amount": float(r["amount"]),
                    "desc": None if pd.isna(r["description"]) else str(r["description"]),
                }
            )
            inserted += 1

    return {"message": "Upload successful", "rows_inserted": inserted}

@app.get("/subscriptions")
def get_subscriptions():
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT date, merchant, amount FROM transactions")).mappings().all()

    if not rows:
        return {"subscriptions": [], "note": "Upload a CSV first."}

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["merchant", "date"])

    subs = []
    for merchant, g in df.groupby("merchant"):
        dates = list(g["date"])
        if len(dates) < 3:
            continue

        gaps = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
        cadence = detect_cadence(gaps)
        if cadence is None:
            continue

        avg_amount = float(g["amount"].abs().tail(3).mean())
        last_paid = dates[-1].to_pydatetime()
        next_charge = predict_next(last_paid, cadence)

        yearly_mult = 12 if cadence == "monthly" else 52 if cadence == "weekly" else 1

        subs.append({
            "merchant": merchant,
            "cadence": cadence,
            "avg_amount": round(avg_amount, 2),
            "last_paid": last_paid.strftime("%Y-%m-%d"),
            "next_charge_est": next_charge.strftime("%Y-%m-%d"),
            "annual_cost_est": round(avg_amount * yearly_mult, 2)
        })

    subs.sort(key=lambda x: x["annual_cost_est"], reverse=True)
    return {"subscriptions": subs, "count": len(subs)}
