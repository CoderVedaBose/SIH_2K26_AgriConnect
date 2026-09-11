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
import math
import itertools
import urllib.parse
import urllib.request
import re

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
def get_forecast(
    region: Optional[str] = None,
    crop: Optional[str] = None,
    forecast_date: Optional[str] = None,
):
    """
    Generate a demand prediction using the trained model.

    Parameters:
        region: Selected region
        crop: Selected crop
        forecast_date: Selected future month/date from the frontend.
                      Accepts YYYY-MM or YYYY-MM-DD.

    If forecast_date is not supplied, the backend keeps the old behavior
    and forecasts the month immediately after the latest historical record.
    """

    if demand_preprocessor is None or demand_model is None or demand_df is None:
        raise HTTPException(
            status_code=500,
            detail="Demand forecasting model is not loaded."
        )

    # ------------------------------------------------------------
    # 1. Find valid Region/Crop combinations
    # ------------------------------------------------------------
    groups = (
        demand_df.groupby(["Region", "Crop"])
        .size()
        .reset_index(name="records")
    )

    groups = groups[groups["records"] >= 12]

    # Keep the existing default behavior if no region/crop is supplied.
    if not region and not crop:
        if groups.empty:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No Region/Crop combination has at least "
                    "12 historical records."
                )
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

    # ------------------------------------------------------------
    # 2. Get historical data for selected Region + Crop
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # 3. Determine FORECAST DATE
    # ------------------------------------------------------------
    last_date = series["Date"].max()

    if forecast_date:
        try:
            # Accept YYYY-MM
            if re.fullmatch(r"\d{4}-\d{2}", forecast_date):
                future_date = pd.to_datetime(
                    forecast_date + "-01"
                )

            # Accept YYYY-MM-DD
            elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", forecast_date):
                future_date = pd.to_datetime(forecast_date)

            else:
                raise ValueError(
                    "Date must be in YYYY-MM or YYYY-MM-DD format."
                )

        except Exception:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid forecast_date. "
                    "Use YYYY-MM or YYYY-MM-DD."
                )
            )

        # No restriction on which month/year can be picked — past,
        # present, or future are all accepted. The model always uses
        # the most recent 12 historical observations as its lag
        # features (see create_forecast_features), so every date
        # still produces a real model prediction.

    else:
        # Preserve the original behavior when no date is selected.
        future_date = last_date + pd.DateOffset(months=1)

    # ------------------------------------------------------------
    # 4. Create model features using the SELECTED forecast date
    # ------------------------------------------------------------
    try:
        features = create_forecast_features(
            history,
            region,
            crop,
            future_date
        )

        transformed_features = demand_preprocessor.transform(
            features
        )

        prediction = float(
            demand_model.predict(transformed_features)[0]
        )

    except Exception as exc:
        print("Demand forecasting error:", exc)

        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {exc}"
        )

    prediction = max(0.0, prediction)

    # ------------------------------------------------------------
    # 5. Reliability based on historical data
    # ------------------------------------------------------------
    record_count = len(history)

    if record_count >= 30:
        reliability = 95
    elif record_count >= 24:
        reliability = 92
    elif record_count >= 18:
        reliability = 88
    else:
        reliability = 84

    # ------------------------------------------------------------
    # 6. Return forecast
    # ------------------------------------------------------------
    return {
        "crop": crop,
        "region": region,
        "expected_demand": round(prediction, 2),
        "confidence": reliability,
        "forecast_month": future_date.strftime("%B %Y"),
        "forecast_date": future_date.strftime("%Y-%m-%d"),
        "historical_records": record_count,
        "recommendation": (
            f"Plan production and inventory for approximately "
            f"{round(prediction):,} units of {crop} for "
            f"{future_date.strftime('%B %Y')}."
        )
    }

# ============================================================
# SMART ROUTE OPTIMIZATION
# Real road-network routing using OpenStreetMap + OSRM
# and TSP-style stop ordering with 2-opt improvement.
# ============================================================

GEOCODING_URL = "https://nominatim.openstreetmap.org/search"
OSRM_BASE_URL = "https://router.project-osrm.org"
GEOCODER_USER_AGENT = "AgriConnect-SIH-Demo/1.0"


def _http_get_json(url: str, timeout: int = 15):
    """Small standard-library HTTP helper; avoids another pip dependency."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": GEOCODER_USER_AGENT, "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Map routing service is unavailable right now: {exc}"
        )


def geocode_place(place: str) -> tuple[float, float, str]:
    """Convert a place/address to latitude/longitude using Nominatim.

    The UI can also accept a coordinate pair directly: '22.5726, 88.3639'.
    """
    place = place.strip()
    if not place:
        raise HTTPException(status_code=400, detail="Location cannot be empty.")

    coordinate_match = re.match(
        r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$",
        place,
    )
    if coordinate_match:
        lat = float(coordinate_match.group(1))
        lon = float(coordinate_match.group(2))
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise HTTPException(status_code=400, detail=f"Invalid coordinates: {place}")
        return lat, lon, place

    params = urllib.parse.urlencode({
        "q": place,
        "format": "jsonv2",
        "limit": 1,
        "countrycodes": "in",
    })
    results = _http_get_json(f"{GEOCODING_URL}?{params}")
    if not results:
        # Try a second search without restricting to India, useful if the user
        # enters an address outside the demo region.
        params = urllib.parse.urlencode({"q": place, "format": "jsonv2", "limit": 1})
        results = _http_get_json(f"{GEOCODING_URL}?{params}")

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"Could not find '{place}'. Try a fuller place name or enter coordinates as 'lat, lon'."
        )

    item = results[0]
    return float(item["lat"]), float(item["lon"]), item.get("display_name", place)


def osrm_table(coordinates: list[tuple[float, float]]) -> tuple[list[list[float]], list[list[float]]]:
    """Return road-network duration (seconds) and distance (metres) matrices."""
    coordinate_string = ";".join(f"{lon},{lat}" for lat, lon in coordinates)
    url = f"{OSRM_BASE_URL}/table/v1/driving/{coordinate_string}?annotations=duration,distance"
    data = _http_get_json(url)
    if data.get("code") != "Ok":
        raise HTTPException(
            status_code=503,
            detail=f"OSRM could not build a road network for these locations: {data.get('message', 'unknown error')}"
        )
    return data["durations"], data["distances"]


def osrm_route(coordinates: list[tuple[float, float]]) -> dict:
    """Get the actual road geometry and summary for an ordered route."""
    coordinate_string = ";".join(f"{lon},{lat}" for lat, lon in coordinates)
    url = (
        f"{OSRM_BASE_URL}/route/v1/driving/{coordinate_string}"
        "?overview=full&geometries=geojson&steps=false"
    )
    data = _http_get_json(url)
    if data.get("code") != "Ok" or not data.get("routes"):
        raise HTTPException(
            status_code=503,
            detail=f"OSRM could not calculate the final route: {data.get('message', 'unknown error')}"
        )
    return data["routes"][0]


def route_cost(order: list[int], distance_matrix: list[list[float]]) -> float:
    return sum(distance_matrix[order[i]][order[i + 1]] for i in range(len(order) - 1))


def nearest_neighbor_order(distance_matrix: list[list[float]]) -> list[int]:
    """Build a fast initial route starting at node 0."""
    n = len(distance_matrix)
    unvisited = set(range(1, n))
    order = [0]
    while unvisited:
        current = order[-1]
        nxt = min(unvisited, key=lambda j: distance_matrix[current][j] if distance_matrix[current][j] is not None else float("inf"))
        order.append(nxt)
        unvisited.remove(nxt)
    return order


def two_opt(order: list[int], distance_matrix: list[list[float]]) -> list[int]:
    """Improve an open TSP route while keeping the origin fixed."""
    best = order[:]
    best_cost = route_cost(best, distance_matrix)
    improved = True
    while improved:
        improved = False
        # Do not move the starting point (index 0).
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                candidate_cost = route_cost(candidate, distance_matrix)
                if candidate_cost + 0.001 < best_cost:
                    best = candidate
                    best_cost = candidate_cost
                    improved = True
                    break
            if improved:
                break
    return best


def optimize_stop_order(distance_matrix: list[list[float]]) -> list[int]:
    """Exact open-TSP for small inputs; scalable NN + 2-opt for larger inputs."""
    n = len(distance_matrix)
    destination_count = n - 1
    if destination_count <= 1:
        return list(range(n))

    # Exact TSP path is practical for a small SIH demo (up to 8 stops).
    if destination_count <= 8:
        best_order = None
        best_cost = float("inf")
        for permutation in itertools.permutations(range(1, n)):
            candidate = [0, *permutation]
            cost = route_cost(candidate, distance_matrix)
            if cost < best_cost:
                best_cost = cost
                best_order = candidate
        return best_order

    return two_opt(nearest_neighbor_order(distance_matrix), distance_matrix)



# ============================================================
# REAL ROAD ROUTING + TSP / 2-OPT
# ============================================================
# The route optimizer deliberately uses public services:
#   1) Nominatim (OpenStreetMap) -> place/address -> coordinates
#   2) OSRM -> real road-network distance/time matrix
#   3) TSP + 2-opt -> chooses the order of destinations
#   4) OSRM route -> returns the final real-road geometry for the map
#
# No route values are simulated or randomly generated.

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_TABLE_URL = "https://router.project-osrm.org/table/v1/driving/"
OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving/"

# Small in-process cache prevents repeated geocoding of the same place
# while the FastAPI server is running.
_GEOCODE_CACHE = {}


def _http_get_json(url: str, timeout: int = 20):
    """GET JSON with a proper application User-Agent."""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "AgriConnect-SIH-RouteOptimizer/1.0 "
                          "(educational project)"
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to contact the mapping service. "
                "Please check your internet connection and try again."
            ),
        ) from exc


def geocode_place(place: str):
    """
    Convert a user-entered place/address into (latitude, longitude, label).

    Nominatim is queried with India as the country restriction because the
    application is intended for the Indian agricultural use case.
    """
    key = place.strip().casefold()

    if key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[key]

    query = urllib.parse.urlencode(
        {
            "q": f"{place.strip()}, India",
            "format": "jsonv2",
            "limit": 1,
            "countrycodes": "in",
            "addressdetails": 1,
        }
    )

    data = _http_get_json(
        f"{NOMINATIM_URL}?{query}",
        timeout=20,
    )

    if not data:
        raise HTTPException(
            status_code=400,
            detail=f"Could not find '{place}'. Try a more specific city, district or address.",
        )

    try:
        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Mapping service returned an invalid location for '{place}'.",
        ) from exc

    result = (lat, lon, data[0].get("display_name", place.strip()))
    _GEOCODE_CACHE[key] = result
    return result


def _validate_osrm_matrix(matrix, name: str, n: int):
    if not isinstance(matrix, list) or len(matrix) != n:
        raise HTTPException(
            status_code=502,
            detail=f"OSRM returned an invalid {name} matrix."
        )

    for row in matrix:
        if not isinstance(row, list) or len(row) != n:
            raise HTTPException(
                status_code=502,
                detail=f"OSRM returned an invalid {name} matrix."
            )


def osrm_table(coordinates):
    """
    Get an NxN real-road matrix from OSRM.

    coordinates are [(lat, lon), ...].
    OSRM expects lon,lat in its URL.
    """
    if len(coordinates) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least two locations are required."
        )

    coordinate_string = ";".join(
        f"{lon:.6f},{lat:.6f}"
        for lat, lon in coordinates
    )

    url = (
        f"{OSRM_TABLE_URL}{coordinate_string}"
        "?annotations=duration,distance"
    )

    data = _http_get_json(url, timeout=30)

    if data.get("code") != "Ok":
        raise HTTPException(
            status_code=400,
            detail=(
                "OSRM could not build a road network between all "
                "locations. Please check the entered places."
            ),
        )

    durations = data.get("durations")
    distances = data.get("distances")

    _validate_osrm_matrix(durations, "duration", len(coordinates))
    _validate_osrm_matrix(distances, "distance", len(coordinates))

    # OSRM can return null where no route exists.
    for i in range(len(coordinates)):
        for j in range(len(coordinates)):
            if i != j and (
                durations[i][j] is None or distances[i][j] is None
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "There is no drivable road route between "
                        "at least two of the entered locations."
                    ),
                )

    return durations, distances


def osrm_route(coordinates):
    """Return final real-road route geometry plus distance/time."""
    coordinate_string = ";".join(
        f"{lon:.6f},{lat:.6f}"
        for lat, lon in coordinates
    )

    url = (
        f"{OSRM_ROUTE_URL}{coordinate_string}"
        "?overview=full&geometries=geojson&steps=false"
    )

    data = _http_get_json(url, timeout=30)

    if data.get("code") != "Ok" or not data.get("routes"):
        raise HTTPException(
            status_code=400,
            detail="OSRM could not calculate the final road route."
        )

    route = data["routes"][0]
    geometry = route.get("geometry")

    if not geometry or geometry.get("type") != "LineString":
        raise HTTPException(
            status_code=502,
            detail="OSRM did not return usable road geometry."
        )

    return {
        "distance": float(route["distance"]),
        "duration": float(route["duration"]),
        "geometry": geometry,
    }


def route_cost(order, matrix):
    """Cost of an open route; it does not return to the origin."""
    total = 0.0

    for a, b in zip(order, order[1:]):
        value = matrix[a][b]

        if value is None:
            return math.inf

        total += float(value)

    return total


def _two_opt(order, matrix):
    """
    Improve a route by reversing destination segments.

    The origin (index 0) is kept fixed. Unlike a simplistic distance
    formula, every candidate is evaluated against OSRM's actual road
    network matrix.
    """
    best = order[:]
    best_cost = route_cost(best, matrix)
    improved = True

    while improved:
        improved = False

        # i starts at 1 so index 0 (origin) never moves.
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = (
                    best[:i]
                    + best[i:j + 1][::-1]
                    + best[j + 1:]
                )

                candidate_cost = route_cost(candidate, matrix)

                if candidate_cost + 1e-9 < best_cost:
                    best = candidate
                    best_cost = candidate_cost
                    improved = True
                    break

            if improved:
                break

    return best


def optimize_stop_order(distances):
    """
    Fixed-origin open TSP.

    For small routes (<= 9 destinations), evaluate every possible order,
    giving the exact shortest-distance order for the OSRM matrix.

    For larger routes, use nearest-neighbour followed by 2-opt so the
    dashboard remains responsive.
    """
    n = len(distances)

    if n <= 1:
        return list(range(n))

    destination_count = n - 1

    if destination_count <= 9:
        best_order = None
        best_cost = math.inf

        for permutation in itertools.permutations(range(1, n)):
            candidate = [0] + list(permutation)
            cost = route_cost(candidate, distances)

            if cost < best_cost:
                best_cost = cost
                best_order = candidate

        return best_order or list(range(n))

    # Greedy starting solution.
    remaining = set(range(1, n))
    order = [0]

    while remaining:
        current = order[-1]
        next_stop = min(
            remaining,
            key=lambda idx: (
                float("inf")
                if distances[current][idx] is None
                else float(distances[current][idx])
            ),
        )
        order.append(next_stop)
        remaining.remove(next_stop)

    return _two_opt(order, distances)



@app.post("/optimize-route")
def optimize_route(request: RouteRequest):
    """
    Optimize a delivery route using real geocoding + OSRM road data.

    Input:
      origin: starting place/address
      destinations: comma-separated places from the frontend

    Output includes:
      - real road distance/time
      - optimized stop order
      - baseline vs optimized savings
      - final GeoJSON road geometry for Leaflet
    """
    origin = request.origin.strip()

    # Remove blanks and duplicates while preserving entered order.
    destinations = []
    seen = set()

    for destination in request.destinations:
        value = destination.strip()
        key = value.casefold()

        if value and key not in seen:
            destinations.append(value)
            seen.add(key)

    if not origin:
        raise HTTPException(
            status_code=400,
            detail="Starting point is required."
        )

    if not destinations:
        raise HTTPException(
            status_code=400,
            detail="At least one destination is required."
        )

    # Prevent an accidentally huge public-API request.
    if len(destinations) > 15:
        raise HTTPException(
            status_code=400,
            detail="Please use at most 15 destinations."
        )

    places = [origin] + destinations

    # 1. Real-world geocoding.
    geocoded = [geocode_place(place) for place in places]
    coordinates = [
        (lat, lon)
        for lat, lon, _display_name in geocoded
    ]

    # 2. Real road-network matrix from OSRM.
    durations, distances = osrm_table(coordinates)

    # 3. User-entered order is the baseline.
    original_order = list(range(len(places)))
    original_distance_m = route_cost(original_order, distances)
    original_time_s = route_cost(original_order, durations)

    if not math.isfinite(original_distance_m):
        raise HTTPException(
            status_code=400,
            detail="The entered order contains an unreachable road segment."
        )

    # 4. Solve fixed-origin open TSP and improve it with 2-opt.
    optimized_order = optimize_stop_order(distances)

    if not optimized_order:
        raise HTTPException(
            status_code=500,
            detail="Could not determine an optimized route."
        )

    optimized_matrix_distance_m = route_cost(
        optimized_order,
        distances,
    )

    if not math.isfinite(optimized_matrix_distance_m):
        raise HTTPException(
            status_code=400,
            detail="No complete drivable route exists between the locations."
        )

    # 5. Ask OSRM for the actual final road geometry.
    optimized_coordinates = [
        coordinates[i]
        for i in optimized_order
    ]

    final_route = osrm_route(optimized_coordinates)

    distance_km = final_route["distance"] / 1000.0
    time_minutes = final_route["duration"] / 60.0

    # Compare against the exact same OSRM road-network baseline.
    distance_saved_m = max(
        0.0,
        original_distance_m - final_route["distance"]
    )

    savings_percentage = (
        distance_saved_m / original_distance_m * 100.0
        if original_distance_m > 0
        else 0.0
    )

    # Transparent estimate only — not vehicle telemetry.
    fuel_efficiency_kmpl = 15.0

    baseline_fuel_l = (
        original_distance_m / 1000.0
    ) / fuel_efficiency_kmpl

    optimized_fuel_l = (
        final_route["distance"] / 1000.0
    ) / fuel_efficiency_kmpl

    fuel_saved_l = max(
        0.0,
        baseline_fuel_l - optimized_fuel_l
    )

    route_names = [
        places[i]
        for i in optimized_order
    ]

    return {
        "distance": round(distance_km, 2),
        "time": round(time_minutes),
        "fuel_used": round(optimized_fuel_l, 2),
        "fuel_saved": round(savings_percentage, 1),
        "fuel_saved_litres": round(fuel_saved_l, 2),

        "route": " → ".join(route_names),
        "ordered_stops": route_names,

        "algorithm": (
            "Fixed-origin TSP + 2-opt on OSRM road-network costs"
        ),
        "routing_engine": "OpenStreetMap Nominatim + OSRM",

        "baseline_distance": round(
            original_distance_m / 1000.0, 2
        ),
        "baseline_time": round(
            original_time_s / 60.0
        ),
        "distance_saved": round(
            distance_saved_m / 1000.0, 2
        ),
        "savings_percentage": round(
            savings_percentage, 1
        ),

        "geometry": final_route["geometry"],

        "locations": [
            {
                "name": places[i],
                "latitude": coordinates[i][0],
                "longitude": coordinates[i][1],
            }
            for i in optimized_order
        ],
    }