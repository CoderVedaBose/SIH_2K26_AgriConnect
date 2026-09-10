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
import math
import heapq

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

EMPTY_DB = {"products": SEED_PRODUCTS, "orders": [], "farmers": []}


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
# SMART ROUTE OPTIMIZATION
# ============================================================

# Coordinates used by the route optimizer.  The frontend accepts
# these names; adding another place only requires adding its coordinates.
LOCATION_COORDINATES = {
    # Chhattisgarh
    "Raipur": (21.2514, 81.6296),
    "Durg": (21.1904, 81.2849),
    "Bhilai": (21.1938, 81.3509),
    "Balod": (20.7308, 81.2054),
    "Bemetara": (21.7156, 81.5340),
    "Rajnandgaon": (21.0972, 81.0287),
    "Bilaspur": (22.0797, 82.1409),
    "Korba": (22.3595, 82.7501),
    "Mahasamund": (21.1092, 82.0973),
    "Dhamtari": (20.7074, 81.5498),
    "Jagdalpur": (19.0748, 82.0090),
    "Kanker": (20.2719, 81.4917),
    "Kawardha": (22.0085, 81.2244),
    "Janjgir": (21.9700, 82.5800),
    "Raigarh": (21.8974, 83.3950),
    "Ambikapur": (23.1355, 83.1811),

    # West Bengal locations already used by AgriConnect seed products
    "Kolkata": (22.5726, 88.3639),
    "Nadia": (23.4058, 88.5245),
    "Hooghly": (22.8956, 88.4025),
    "Howrah": (22.5958, 88.2636),
    "Malda": (25.0108, 88.1411),
    "Burdwan": (23.2324, 87.8615),
    "Barasat": (22.7215, 88.4829),
    "Birbhum": (23.8402, 87.6186),
}


def normalize_location_name(location: str) -> str:
    """Normalize common user-entered location variations."""
    value = " ".join(location.strip().split())

    aliases = {
        "Raipur, CG": "Raipur",
        "Raipur, Chhattisgarh": "Raipur",
        "Durg, CG": "Durg",
        "Bhilai, CG": "Bhilai",
        "Nadia, WB": "Nadia",
        "Hooghly, WB": "Hooghly",
        "Howrah, WB": "Howrah",
        "Malda, WB": "Malda",
        "Burdwan, WB": "Burdwan",
        "Barasat, WB": "Barasat",
        "Birbhum, WB": "Birbhum",
        "Kolkata, WB": "Kolkata",
    }

    return aliases.get(value, value)


def haversine_distance(coord1, coord2) -> float:
    """Return geographical distance between two coordinates in km."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    earth_radius_km = 6371.0

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    return earth_radius_km * 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a),
    )


def build_location_graph(locations: list[str]) -> dict[str, list[tuple[str, float]]]:
    """Build a weighted geographical graph from the supplied locations."""
    graph = {location: [] for location in locations}

    for i, location_a in enumerate(locations):
        for location_b in locations[i + 1:]:
            distance = haversine_distance(
                LOCATION_COORDINATES[location_a],
                LOCATION_COORDINATES[location_b],
            )
            graph[location_a].append((location_b, distance))
            graph[location_b].append((location_a, distance))

    return graph


def dijkstra(graph: dict, start: str, end: str) -> tuple[float, list[str]]:
    """Return shortest distance and path from start to end."""
    distances = {node: float("inf") for node in graph}
    previous = {node: None for node in graph}
    distances[start] = 0.0

    queue = [(0.0, start)]

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_distance > distances[current_node]:
            continue

        if current_node == end:
            break

        for neighbor, weight in graph[current_node]:
            candidate = current_distance + weight

            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = current_node
                heapq.heappush(queue, (candidate, neighbor))

    if distances[end] == float("inf"):
        return float("inf"), []

    path = []
    current = end

    while current is not None:
        path.append(current)
        current = previous[current]

    path.reverse()
    return distances[end], path


@app.get("/route/options")
def get_route_options():
    """Return supported locations for the route UI."""
    return {
        "locations": sorted(LOCATION_COORDINATES.keys()),
        "algorithm": "Dijkstra + Haversine",
    }


@app.post("/optimize-route")
def optimize_route(request: RouteRequest):
    """Optimize a multi-stop delivery route using Dijkstra's algorithm."""
    if not request.origin.strip():
        raise HTTPException(
            status_code=400,
            detail="Starting point is required.",
        )

    if not request.destinations:
        raise HTTPException(
            status_code=400,
            detail="At least one destination is required.",
        )

    origin = normalize_location_name(request.origin)
    destinations = [
        normalize_location_name(destination)
        for destination in request.destinations
        if destination.strip()
    ]

    # Remove duplicates and don't treat the origin as a delivery stop.
    destinations = list(dict.fromkeys(destinations))
    destinations = [d for d in destinations if d != origin]

    if not destinations:
        raise HTTPException(
            status_code=400,
            detail="Choose at least one destination different from the origin.",
        )

    locations = [origin] + destinations

    unknown = [
        location
        for location in locations
        if location not in LOCATION_COORDINATES
    ]

    if unknown:
        supported = ", ".join(sorted(LOCATION_COORDINATES.keys()))
        raise HTTPException(
            status_code=400,
            detail=(
                "Location not supported: "
                + ", ".join(unknown)
                + ". Please choose a location from the supported list. "
                + "Supported locations: "
                + supported
            ),
        )

    graph = build_location_graph(locations)

    # Greedy multi-stop sequencing, with Dijkstra used for every leg.
    # This keeps the implementation lightweight while each leg is an
    # actual shortest-path calculation rather than a random simulation.
    current = origin
    remaining = destinations.copy()
    optimized_route = [origin]
    total_distance = 0.0

    while remaining:
        best_destination = None
        best_distance = float("inf")
        best_path = []

        for destination in remaining:
            distance, path = dijkstra(graph, current, destination)

            if distance < best_distance:
                best_distance = distance
                best_destination = destination
                best_path = path

        if best_destination is None or not best_path:
            raise HTTPException(
                status_code=500,
                detail="Could not calculate a route between the selected locations.",
            )

        optimized_route.extend(best_path[1:])
        total_distance += best_distance
        current = best_destination
        remaining.remove(best_destination)

    # Compare against the exact order entered by the user.
    original_route = [origin] + destinations
    original_distance = 0.0

    for i in range(len(original_route) - 1):
        original_distance += haversine_distance(
            LOCATION_COORDINATES[original_route[i]],
            LOCATION_COORDINATES[original_route[i + 1]],
        )

    distance_saved = max(0.0, original_distance - total_distance)
    savings_percentage = (
        (distance_saved / original_distance) * 100
        if original_distance > 0
        else 0.0
    )

    # Practical estimates for a small agricultural delivery vehicle.
    average_speed_kmh = 40.0
    mileage_km_per_litre = 18.0

    time_minutes = round((total_distance / average_speed_kmh) * 60)
    fuel_used = total_distance / mileage_km_per_litre

    return {
        "distance": round(total_distance, 2),
        "time": time_minutes,
        "fuel_used": round(fuel_used, 2),
        "fuel_saved": round(distance_saved / mileage_km_per_litre, 2),
        "original_distance": round(original_distance, 2),
        "distance_saved": round(distance_saved, 2),
        "savings_percentage": round(savings_percentage, 1),
        "route": " → ".join(optimized_route),
        "stops": optimized_route,
        "algorithm": "Dijkstra + Haversine",
        "average_speed": average_speed_kmh,
        "message": "Route calculated from geographical coordinates using Dijkstra's shortest-path algorithm.",
    }
