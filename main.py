"""
AgriConnect – FastAPI Backend
==============================

A complete, self-contained backend for the AgriConnect frontend
(index.html / script.js / style.css).

Storage: a plain JSON file called `database.json` sitting next to this
script. No external database, no ORM, no setup — just run this file
and `database.json` is created (and updated) automatically.

Run it with:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

The frontend already points at http://127.0.0.1:8000 (see API_BASE_URL
in script.js), so once this is running you can just open index.html
(or serve it with any static server) and everything will connect.
"""

from __future__ import annotations

import json
import random
import threading
import hashlib

import joblib
import numpy as np
from xgboost import XGBRegressor
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"
DB_FILE = BACKEND_DIR / "database.json"
PREPROCESSOR_FILE = BACKEND_DIR / "demand_preprocessor.pkl"
XGB_MODEL_FILE = BACKEND_DIR / "demand_xgb_model.json"
DATA_FILE = BACKEND_DIR / "Crop-Demand-Data.csv"

# Real ML demand forecasting assets
demand_preprocessor = None
demand_model = None
demand_df = None


def load_demand_forecasting_assets() -> None:
    """Load the separately saved preprocessor and native XGBoost model."""
    global demand_preprocessor, demand_model, demand_df

    if not PREPROCESSOR_FILE.exists():
        raise FileNotFoundError(
            f"Forecasting preprocessor not found: {PREPROCESSOR_FILE}"
        )

    if not XGB_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Forecasting XGBoost model not found: {XGB_MODEL_FILE}"
        )

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Forecasting dataset not found: {DATA_FILE}"
        )

    demand_preprocessor = joblib.load(PREPROCESSOR_FILE)

    demand_model = XGBRegressor()
    demand_model.load_model(str(XGB_MODEL_FILE))

    demand_df = pd.read_csv(DATA_FILE)

    required_columns = {
        "Year", "Month", "Region", "Crop", "Market_Demand"
    }
    missing = required_columns - set(demand_df.columns)
    if missing:
        raise ValueError(
            "Crop-Demand-Data.csv is missing columns: "
            + ", ".join(sorted(missing))
        )

    demand_df["Year"] = pd.to_numeric(demand_df["Year"], errors="coerce")
    demand_df["Month"] = pd.to_numeric(demand_df["Month"], errors="coerce")
    demand_df["Region"] = demand_df["Region"].astype(str).str.strip()
    demand_df["Crop"] = demand_df["Crop"].astype(str).str.strip()
    demand_df["Market_Demand"] = pd.to_numeric(
        demand_df["Market_Demand"], errors="coerce"
    )

    demand_df["Date"] = pd.to_datetime(
        dict(
            year=demand_df["Year"],
            month=demand_df["Month"],
            day=1,
        ),
        errors="coerce",
    )

    demand_df.dropna(
        subset=["Date", "Market_Demand", "Region", "Crop"],
        inplace=True,
    )
    demand_df.sort_values(
        ["Region", "Crop", "Date"], inplace=True
    )
    demand_df.reset_index(drop=True, inplace=True)


def create_forecast_features(history, region, crop, future_date):
    """Create exactly the feature columns used during model training."""
    if len(history) < 12:
        raise ValueError(
            "At least 12 historical observations are required "
            "for lag-12 forecasting."
        )

    month_num = future_date.month
    quarter = future_date.quarter
    year_num = future_date.year

    return pd.DataFrame([{
        "Region": region,
        "Crop": crop,
        "month_num": month_num,
        "quarter": quarter,
        "year_num": year_num,
        "month_sin": np.sin(2 * np.pi * month_num / 12),
        "month_cos": np.cos(2 * np.pi * month_num / 12),
        "lag_1": history[-1],
        "lag_2": history[-2],
        "lag_3": history[-3],
        "lag_6": history[-6],
        "lag_12": history[-12],
        "rolling_mean_3": float(np.mean(history[-3:])),
        "rolling_mean_6": float(np.mean(history[-6:])),
    }])


# Guards concurrent read/modify/write cycles on database.json
db_lock = threading.Lock()


# ============================================================
# SEED DATA
# (Mirrors the demo products already hard-coded in script.js so the
#  storefront looks identical the moment the real backend takes over)
# ============================================================

SEED_PRODUCTS = [
    {"id": 1, "name": "Fresh Tomatoes", "category": "Vegetables", "price": 42,
     "unit": "kg", "farmer": "Green Valley Farms", "location": "Nadia, WB",
     "rating": 4.8, "badge": "High Demand"},
    {"id": 2, "name": "Organic Potatoes", "category": "Vegetables", "price": 32,
     "unit": "kg", "farmer": "Maa Kali FPO", "location": "Hooghly, WB",
     "rating": 4.7, "badge": "Organic"},
    {"id": 3, "name": "Fresh Mangoes", "category": "Fruits", "price": 95,
     "unit": "kg", "farmer": "Mango Valley", "location": "Malda, WB",
     "rating": 4.9, "badge": "Fresh Today"},
    {"id": 4, "name": "Basmati Rice", "category": "Grains", "price": 88,
     "unit": "kg", "farmer": "Bengal FPO", "location": "Burdwan, WB",
     "rating": 4.8, "badge": "Best Seller"},
    {"id": 5, "name": "Red Onions", "category": "Vegetables", "price": 38,
     "unit": "kg", "farmer": "Krishi Shakti FPO", "location": "Nadia, WB",
     "rating": 4.6, "badge": "High Demand"},
    {"id": 6, "name": "Fresh Milk", "category": "Dairy", "price": 58,
     "unit": "litre", "farmer": "Pure Dairy Farms", "location": "Barasat, WB",
     "rating": 4.9, "badge": "Farm Fresh"},
    {"id": 7, "name": "Organic Wheat", "category": "Organic", "price": 65,
     "unit": "kg", "farmer": "Organic India FPO", "location": "Birbhum, WB",
     "rating": 4.8, "badge": "Organic"},
    {"id": 8, "name": "Green Chillies", "category": "Spices", "price": 75,
     "unit": "kg", "farmer": "Fresh Harvest", "location": "Howrah, WB",
     "rating": 4.7, "badge": "Fresh Today"},
]

EMPTY_DB = {"products": SEED_PRODUCTS, "orders": [], "farmers": [], "buyers": []}


# ============================================================
# JSON "DATABASE" HELPERS
# ============================================================

def init_db() -> None:
    """Create database.json with seed data if it doesn't exist yet."""
    if not DB_FILE.exists():
        DB_FILE.write_text(json.dumps(EMPTY_DB, indent=2), encoding="utf-8")


def read_db() -> dict:
    with db_lock:
        if not DB_FILE.exists():
            init_db()
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)


def write_db(data: dict) -> None:
    with db_lock:
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def next_id(items: list) -> int:
    return (max((item["id"] for item in items), default=0)) + 1


# ============================================================
# PYDANTIC MODELS
# ============================================================

class Product(BaseModel):
    id: int
    name: str
    category: str
    price: float
    unit: str
    farmer: str
    location: str
    rating: float = 4.5
    badge: Optional[str] = None


class ProductCreate(BaseModel):
    name: str
    category: str
    price: float
    unit: str
    farmer: str
    location: str
    rating: float = 4.5
    badge: Optional[str] = None


class OrderItem(BaseModel):
    product_id: int
    quantity: int = Field(gt=0)


class OrderCreate(BaseModel):
    customer_name: str
    phone: str
    address: str
    payment_method: str
    items: List[OrderItem]


class Order(OrderCreate):
    id: int
    total: float
    status: str = "Placed"
    created_at: str


class FarmerCreate(BaseModel):
    name: str
    phone: str
    location: str
    produce: str


class Farmer(FarmerCreate):
    id: int
    created_at: str


class FarmerAuthCreate(FarmerCreate):
    password: str = Field(min_length=6)


class LoginRequest(BaseModel):
    phone: str
    password: str


class RouteRequest(BaseModel):
    origin: str
    destinations: List[str]


# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(
    title="AgriConnect API",
    description="Backend for the AgriConnect farmer-direct marketplace",
    version="1.0.0",
)

# Allow the frontend (opened as a local file, or served from any port/
# Live Server instance) to call this API without CORS errors.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    load_demand_forecasting_assets()


# ============================================================
# ROOT / HEALTH
# ============================================================

@app.get("/")
def root():
    return {"status": "ok", "message": "AgriConnect API is running"}


# ============================================================
# PRODUCTS
# ============================================================

@app.get("/products", response_model=List[Product])
def get_products():
    db = read_db()
    return db["products"]


@app.get("/products/{product_id}", response_model=Product)
def get_product(product_id: int):
    db = read_db()
    for p in db["products"]:
        if p["id"] == product_id:
            return p
    raise HTTPException(status_code=404, detail="Product not found")


@app.post("/products", response_model=Product, status_code=201)
def create_product(product: ProductCreate):
    db = read_db()
    new_product = product.model_dump()
    new_product["id"] = next_id(db["products"])
    db["products"].append(new_product)
    write_db(db)
    return new_product


@app.delete("/products/{product_id}")
def delete_product(product_id: int):
    db = read_db()
    remaining = [p for p in db["products"] if p["id"] != product_id]
    if len(remaining) == len(db["products"]):
        raise HTTPException(status_code=404, detail="Product not found")
    db["products"] = remaining
    write_db(db)
    return {"message": "Product deleted"}


# ============================================================
# ORDERS
# ============================================================

@app.get("/orders", response_model=List[Order])
def get_orders():
    db = read_db()
    return db["orders"]


@app.get("/orders/{order_id}", response_model=Order)
def get_order(order_id: int):
    db = read_db()
    for o in db["orders"]:
        if o["id"] == order_id:
            return o
    raise HTTPException(status_code=404, detail="Order not found")


@app.post("/orders", response_model=Order, status_code=201)
def create_order(order: OrderCreate):
    db = read_db()

    # Validate every product exists and compute the total server-side
    product_map = {p["id"]: p for p in db["products"]}
    total = 0.0
    for item in order.items:
        product = product_map.get(item.product_id)
        if not product:
            raise HTTPException(
                status_code=404,
                detail=f"Product with id {item.product_id} not found",
            )
        total += product["price"] * item.quantity

    new_order = order.model_dump()
    new_order["id"] = next_id(db["orders"])
    new_order["total"] = round(total, 2)
    new_order["status"] = "Placed"
    new_order["created_at"] = datetime.utcnow().isoformat()

    db["orders"].append(new_order)
    write_db(db)
    return new_order


# ============================================================
# FARMERS
# ============================================================

@app.get("/farmers", response_model=List[Farmer])
def get_farmers():
    db = read_db()
    return db["farmers"]


@app.post("/farmers", response_model=Farmer, status_code=201)
def register_farmer(farmer: FarmerCreate):
    db = read_db()
    new_farmer = farmer.model_dump()
    new_farmer["id"] = next_id(db["farmers"])
    new_farmer["created_at"] = datetime.utcnow().isoformat()
    db["farmers"].append(new_farmer)
    write_db(db)
    return new_farmer


# ============================================================
# AUTHENTICATION
# ============================================================

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def find_user_by_phone(db: dict, phone: str, role: str):
    phone = phone.strip()
    if role == "farmer":
        for farmer in db.get("farmers", []):
            if farmer.get("phone") == phone:
                return farmer
        return None

    for buyer in db.get("buyers", []):
        if buyer.get("phone") == phone:
            return buyer
    return None


@app.post("/auth/farmer/register")
def auth_register_farmer(farmer: FarmerAuthCreate):
    db = read_db()
    if find_user_by_phone(db, farmer.phone, "farmer"):
        raise HTTPException(status_code=409, detail="A farmer account with this phone number already exists.")

    new_farmer = farmer.model_dump(exclude={"password"})
    new_farmer["id"] = next_id(db.get("farmers", []))
    new_farmer["created_at"] = datetime.utcnow().isoformat()
    new_farmer["password_hash"] = hash_password(farmer.password)
    db.setdefault("farmers", []).append(new_farmer)
    write_db(db)
    return {"message": "Farmer account created successfully", "farmer_id": new_farmer["id"]}


@app.post("/auth/farmer/login")
def auth_login_farmer(login: LoginRequest):
    db = read_db()
    farmer = find_user_by_phone(db, login.phone, "farmer")
    if not farmer or farmer.get("password_hash") != hash_password(login.password):
        raise HTTPException(status_code=401, detail="Invalid farmer phone number or password.")
    return {"role": "farmer", "id": farmer["id"], "name": farmer["name"], "phone": farmer["phone"]}


@app.post("/auth/buyer/register")
def auth_register_buyer(payload: dict):
    db = read_db()
    phone = str(payload.get("phone", "")).strip()
    name = str(payload.get("name", "")).strip()
    password = str(payload.get("password", ""))
    if not name or len(phone) != 10 or not phone.isdigit() or len(password) < 6:
        raise HTTPException(status_code=422, detail="Name, valid 10-digit phone and 6+ character password are required.")
    if find_user_by_phone(db, phone, "buyer"):
        raise HTTPException(status_code=409, detail="A buyer account with this phone number already exists.")
    buyers = db.setdefault("buyers", [])
    buyer = {"id": next_id(buyers), "name": name, "phone": phone, "password_hash": hash_password(password), "created_at": datetime.utcnow().isoformat()}
    buyers.append(buyer)
    write_db(db)
    return {"message": "Buyer account created successfully", "buyer_id": buyer["id"]}


@app.post("/auth/buyer/login")
def auth_login_buyer(login: LoginRequest):
    db = read_db()
    buyer = find_user_by_phone(db, login.phone, "buyer")
    if not buyer or buyer.get("password_hash") != hash_password(login.password):
        raise HTTPException(status_code=401, detail="Invalid buyer phone number or password.")
    return {"role": "buyer", "id": buyer["id"], "name": buyer["name"], "phone": buyer["phone"]}


# ============================================================
# AI DEMAND FORECAST  (GET /forecast)
# ============================================================

@app.get("/forecast/options")
def get_forecast_options():
    """Return only crop/region combinations with enough history."""
    if demand_df is None:
        raise HTTPException(
            status_code=500,
            detail="Demand forecasting model is not loaded."
        )

    groups = (
        demand_df
        .groupby(["Region", "Crop"])
        .size()
        .reset_index(name="records")
    )
    groups = groups[groups["records"] >= 12]

    return {
        "regions": sorted(groups["Region"].unique().tolist()),
        "crops": sorted(groups["Crop"].unique().tolist()),
        "pairs": [
            {
                "region": row["Region"],
                "crop": row["Crop"]
            }
            for _, row in groups.iterrows()
        ]
    }


@app.get("/forecast")
def get_forecast(region: Optional[str] = None, crop: Optional[str] = None):
    """Generate a real next-month demand prediction using the trained model."""
    if demand_preprocessor is None or demand_model is None or demand_df is None:
        raise HTTPException(
            status_code=500,
            detail="Demand forecasting model is not loaded."
        )

    # Keep the endpoint compatible with the existing frontend, which calls
    # /forecast without query parameters. If region/crop are supplied, use
    # that exact combination; otherwise use the first available combination
    # with at least 12 historical records.
    groups = (
        demand_df.groupby(["Region", "Crop"]).size()
        .reset_index(name="records")
    )
    groups = groups[groups["records"] >= 12]

    if not region and not crop:
        if groups.empty:
            raise HTTPException(
                status_code=400,
                detail="No Region/Crop combination has at least 12 historical records."
            )
        region = str(groups.iloc[0]["Region"])
        crop = str(groups.iloc[0]["Crop"])
    elif not region or not crop:
        raise HTTPException(
            status_code=400,
            detail="Provide both region and crop, or provide neither."
        )
    else:
        region = region.strip()
        crop = crop.strip()

    series = demand_df[
        (demand_df["Region"] == region)
        & (demand_df["Crop"] == crop)
    ].sort_values("Date")

    if series.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No demand history found for {crop} in {region}."
        )

    history = series["Market_Demand"].astype(float).tolist()

    if len(history) < 12:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Only {len(history)} historical records are available "
                f"for {crop} in {region}; 12 are required."
            )
        )

    last_date = series["Date"].max()
    future_date = last_date + pd.DateOffset(months=1)

    try:
        features = create_forecast_features(
            history, region, crop, future_date
        )
        transformed_features = demand_preprocessor.transform(features)
        prediction = float(demand_model.predict(transformed_features)[0])
    except Exception as exc:
        print("Demand forecasting error:", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {exc}"
        )

    prediction = max(0.0, prediction)

    # Reliability is based on the amount of historical data, not randomness.
    record_count = len(history)
    if record_count >= 30:
        reliability = 95
    elif record_count >= 24:
        reliability = 92
    elif record_count >= 18:
        reliability = 88
    else:
        reliability = 84

    return {
        "crop": crop,
        "region": region,
        "expected_demand": round(prediction, 2),
        "confidence": reliability,
        "forecast_month": future_date.strftime("%B %Y"),
        "historical_records": record_count,
        "recommendation": (
            f"Plan production and inventory for approximately "
            f"{round(prediction):,} units of {crop} next month."
        )
    }


# ============================================================
# AI ROUTE OPTIMIZATION  (POST /optimize-route)
# ============================================================

@app.post("/optimize-route")
def optimize_route(request: RouteRequest):
    if not request.destinations:
        raise HTTPException(status_code=400, detail="At least one destination is required")

    # Simple simulated "optimization": shuffle destinations deterministically
    # by name length + alphabetical order to produce a plausible-looking
    # optimized route. Swap this out for a real routing/ML service later.
    ordered = sorted(request.destinations, key=lambda d: (len(d), d))

    distance = round(random.uniform(20, 120), 1)
    time_minutes = round(distance * random.uniform(1.2, 1.8))
    fuel_saved = random.randint(10, 25)

    route_str = " → ".join([request.origin] + ordered)

    return {
        "distance": distance,
        "time": time_minutes,
        "fuel_saved": fuel_saved,
        "route": route_str,
    }
