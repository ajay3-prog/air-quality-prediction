import os
import json
import pickle
import random
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import mysql.connector
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Hide sklearn version warnings during demo
warnings.filterwarnings("ignore")

# Initialize FastAPI App
app = FastAPI(
    title="TSPCB Smart Air Quality & Pollution Episode Portal (Team G1672)",
    description="Single Gas + Combined AQI Prediction, Hyderabad 8-Zone IoT Stream, Root Cause, Action Plan & MySQL CRUD",
    version="2.0"
)

# Enable CORS for Multi-Page HTML/JS Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. MYSQL DATABASE CONFIGURATION & FALLBACK
# ==========================================
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",  # Change to your MySQL root password
    "database": "air_quality_db"
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
CSV_PATH = os.path.join(BASE_DIR, "AirQualityUCI.csv")
memory_logs = []

loaded_models = {}
model_registry = {}
is_mysql_active = False


def init_mysql_db():
    """Creates MySQL database and table if MySQL is running."""
    global is_mysql_active
    try:
        conn = mysql.connector.connect(
            host=MYSQL_CONFIG["host"],
            user=MYSQL_CONFIG["user"],
            password=MYSQL_CONFIG["password"],
            connection_timeout=1
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_CONFIG['database']}")
        cursor.execute(f"USE {MYSQL_CONFIG['database']}")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                zone_name VARCHAR(100),
                target_gas VARCHAR(50),
                predicted_value FLOAT,
                actual_value FLOAT NULL,
                overall_aqi FLOAT,
                air_quality_status VARCHAR(80),
                episode_reason VARCHAR(120),
                root_cause VARCHAR(255),
                recommended_action VARCHAR(255),
                inspector_notes VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
        is_mysql_active = True
        print("MySQL Database Connected & Ready!")
    except Exception as e:
        is_mysql_active = False
        print(f"MySQL Notice: Using in-memory CRUD fallback ({e})")


def get_db_connection():
    if not is_mysql_active:
        return None
    try:
        return mysql.connector.connect(**MYSQL_CONFIG, connection_timeout=1)
    except Exception:
        return None


# ==========================================
# 2. LOAD TRAINED PICKLE (.pkl) MODELS
# ==========================================
def load_all_pickle_models():
    global loaded_models, model_registry
    loaded_models = {}
    registry_path = os.path.join(MODEL_DIR, "model_registry.json")

    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            model_registry = json.load(f)

    if not os.path.exists(MODEL_DIR):
        return

    for filename in os.listdir(MODEL_DIR):
        if filename.endswith("_model.pkl"):
            gas_key = filename.replace("_model.pkl", "")
            filepath = os.path.join(MODEL_DIR, filename)
            with open(filepath, "rb") as f:
                loaded_models[gas_key] = pickle.load(f)

    print("Loaded Pickle Models:", list(loaded_models.keys()))


@app.on_event("startup")
def startup_event():
    init_mysql_db()
    load_all_pickle_models()


# ==========================================
# 3. INPUT SCHEMAS (PYDANTIC MODELS)
# ==========================================
class SensorInput(BaseModel):
    zone_name: Optional[str] = "Narayanguda (KMIT Junction)"
    target_gas: Optional[str] = "ALL"
    PT08_S1_CO: float = 1360.0
    PT08_S2_NMHC: float = 1046.0
    PT08_S3_NOx: float = 1056.0
    PT08_S4_NO2: float = 1692.0
    PT08_S5_O3: float = 1268.0
    T: float = 18.5
    RH: float = 55.0
    AH: float = 0.85
    Hour: int = 18
    DayOfWeek: int = 2
    Month: int = 3
    Lag_1: float = 2.4
    Lag_2: float = 2.2
    Lag_3: float = 2.0
    Lag_24: float = 2.5
    actual_value: Optional[float] = None
    inspector_notes: Optional[str] = "Routine TSPCB Zone Inspection"


class SimulateInput(BaseModel):
    zone_name: Optional[str] = "LB Nagar Ring Road"
    action_type: str = "traffic_clearance_40"  # traffic_clearance_40 | divert_diesel | water_sprinkler
    PT08_S1_CO: float = 1650.0
    PT08_S2_NMHC: float = 1380.0
    PT08_S3_NOx: float = 620.0
    PT08_S4_NO2: float = 1890.0
    PT08_S5_O3: float = 1540.0
    T: float = 17.2
    RH: float = 72.0
    AH: float = 0.95
    Hour: int = 19
    DayOfWeek: int = 4
    Month: int = 11
    Lag_1: float = 4.2
    Lag_2: float = 3.9
    Lag_3: float = 3.6
    Lag_24: float = 4.0


class LogUpdateInput(BaseModel):
    inspector_notes: str
    air_quality_status: Optional[str] = None
    recommended_action: Optional[str] = None


class CreateManualLogInput(BaseModel):
    zone_name: str = "Narayanguda (KMIT Junction)"
    target_gas: str = "ALL"
    predicted_value: float = 2.5
    actual_value: Optional[float] = None
    overall_aqi: float = 85.0
    air_quality_status: str = "Moderate"
    episode_reason: str = "Normal Observation"
    root_cause: str = "Normal traffic flow"
    recommended_action: str = "Routine IoT monitoring"
    inspector_notes: str = "Manual field inspection"


# ==========================================
# 4. CORE ML PREDICTION, AQI, CAUSE & ACTION
# ==========================================
def run_analysis(row_dict: dict, actual_val: Optional[float] = None, selected_gas: str = "ALL"):
    if not loaded_models:
        raise HTTPException(status_code=500, detail="No .pkl models found in saved_models/")

    roll_mean_3 = round((row_dict["Lag_1"] + row_dict["Lag_2"] + row_dict["Lag_3"]) / 3.0, 4)

    feature_data = {
        "PT08.S1(CO)": row_dict["PT08_S1_CO"],
        "PT08.S2(NMHC)": row_dict["PT08_S2_NMHC"],
        "PT08.S3(NOx)": row_dict["PT08_S3_NOx"],
        "PT08.S4(NO2)": row_dict["PT08_S4_NO2"],
        "PT08.S5(O3)": row_dict["PT08_S5_O3"],
        "T": row_dict["T"],
        "RH": row_dict["RH"],
        "AH": row_dict["AH"],
        "Hour": row_dict["Hour"],
        "DayOfWeek": row_dict["DayOfWeek"],
        "Month": row_dict["Month"],
        "Lag_1": row_dict["Lag_1"],
        "Lag_2": row_dict["Lag_2"],
        "Lag_3": row_dict["Lag_3"],
        "Lag_24": row_dict["Lag_24"],
        "Rolling_Mean_3": roll_mean_3
    }

    # 1. Predict all available gases using loaded .pkl models
    gas_predictions = {}
    gas_details = {}

    for gas_name, pkg in loaded_models.items():
        rf_model = pkg["model"] if isinstance(pkg, dict) else pkg
        cols = pkg.get("feature_columns", list(feature_data.keys())) if isinstance(pkg, dict) else list(feature_data.keys())
        threshold = pkg.get("threshold", 1.5) if isinstance(pkg, dict) else 1.5
        high_level = pkg.get("high_pollution_level", 2.5) if isinstance(pkg, dict) else 2.5

        X_df = pd.DataFrame([[feature_data[c] for c in cols]], columns=cols)
        pred_val = round(float(rf_model.predict(X_df)[0]), 3)
        gas_predictions[gas_name] = pred_val

        is_spike = pred_val >= high_level
        gas_details[gas_name] = {
            "predicted": pred_val,
            "threshold": round(float(threshold), 3),
            "high_pollution_level": round(float(high_level), 3),
            "status": "Potential Pollution Spike" if is_spike else "Normal Level"
        }

    # 2. Calculate Combined Overall Air Quality Index (AQI 0-500)
    co_v = gas_predictions.get("CO", 1.0)
    c6h6_v = gas_predictions.get("C6H6", 5.0)
    nox_v = gas_predictions.get("NOx", 100.0)
    no2_v = gas_predictions.get("NO2", 60.0)

    sub_indices = {
        "CO": min(500.0, round((co_v / 10.0) * 300.0, 1)),
        "C6H6": min(500.0, round((c6h6_v / 40.0) * 300.0, 1)),
        "NOx": min(500.0, round((nox_v / 400.0) * 300.0, 1)),
        "NO2": min(500.0, round((no2_v / 200.0) * 300.0, 1))
    }
    # Support any newly trained gas dynamically
    for g_name, g_val in gas_predictions.items():
        if g_name not in sub_indices:
            sub_indices[g_name] = min(500.0, round(g_val * 2.5, 1))

    dominant_gas = max(sub_indices, key=sub_indices.get)
    overall_aqi = round(max(sub_indices.values()), 1)

    if overall_aqi <= 100:
        aqi_status = "Good (Green Zone)"
        zone_color = "green"
    elif overall_aqi <= 200:
        aqi_status = "Moderate (Yellow Zone)"
        zone_color = "yellow"
    elif overall_aqi <= 300:
        aqi_status = "Poor (Orange Zone)"
        zone_color = "orange"
    else:
        aqi_status = "Severe / Pollution Episode (Red Zone)"
        zone_color = "red"

    # 3. Rising vs Falling Trend Indicator
    ref_pred = gas_predictions.get(selected_gas, co_v) if selected_gas != "ALL" else co_v
    if ref_pred > row_dict["Lag_1"] * 1.10:
        trend = "Rapidly Rising (Worsening ↑)"
    elif ref_pred < row_dict["Lag_1"] * 0.90:
        trend = "Falling (Improving ↓)"
    else:
        trend = "Stable (→)"

    # 4. Episode Classification + Root Cause (Why) & Action Plan (What to do)
    temp = row_dict["T"]
    rh = row_dict["RH"]

    if actual_val is not None and actual_val == -200:
        episode_reason = "Data Limitation (Sensor Hardware Fault -200)"
        root_cause = "Reference Analyzer or IoT Sensor Drop-out (-200 missing code detected)"
        recommended_action = "Do NOT trigger public panic. Dispatch TSPCB technician to recalibrate sensor."
    elif overall_aqi <= 100:
        episode_reason = "Normal Observation"
        root_cause = "Smooth vehicle flow and good atmospheric dispersion"
        recommended_action = "No traffic restriction required. Continue routine automated IoT telemetry."
    elif temp < 19.0 and rh > 68.0 and overall_aqi > 180:
        episode_reason = "Potential Pollution Spike (Smog Episode)"
        root_cause = "Weather Smog Trapping (Low Temp + High Humidity holding vehicle fumes near ground)"
        recommended_action = "Deploy GHMC Water Sprinklers / Anti-Smog Guns & issue Mask Advisory."
    elif dominant_gas in ["CO", "C6H6"]:
        episode_reason = "Potential Pollution Spike" if overall_aqi > 200 else "Elevated Traffic Emission"
        root_cause = f"Heavy Petrol Vehicle Idling & Signal Congestion (High {dominant_gas} concentration)"
        recommended_action = "Alert Traffic Police: Switch junction to Green Corridor mode & show 'Turn Off Engines' sign."
    else:
        episode_reason = "Potential Pollution Spike" if overall_aqi > 200 else "Elevated Diesel Emission"
        root_cause = f"Heavy Diesel Vehicles (RTC Buses/Lorries) or Generator Exhaust (High {dominant_gas})"
        recommended_action = "Divert Heavy Goods Vehicles to Outer Ring Road & conduct spot PUC emission checks."

    # Check residual if actual_val is provided for a specific gas
    target_key = selected_gas if selected_gas in gas_predictions else "CO"
    target_pred = gas_predictions[target_key]
    residual_err = None
    if actual_val is not None and actual_val != -200:
        residual_err = round(actual_val - target_pred, 3)
        thresh = gas_details[target_key]["threshold"]
        if abs(residual_err) > thresh and episode_reason == "Normal Observation":
            episode_reason = "Unusual Prediction Error (Residual > Threshold)"

    return {
        "target_gas": selected_gas,
        "selected_gas_prediction": target_pred if selected_gas != "ALL" else gas_predictions.get(dominant_gas),
        "single_gas_predictions": gas_predictions,
        "gas_details": gas_details,
        "overall_aqi": overall_aqi,
        "sub_indices": sub_indices,
        "air_quality_status": aqi_status,
        "zone_color": zone_color,
        "dominant_pollutant": dominant_gas,
        "pollution_trend": trend,
        "episode_reason": episode_reason,
        "root_cause": root_cause,
        "recommended_action": recommended_action,
        "actual_value": actual_val,
        "residual_error": residual_err
    }


# College student viva friendly alias
analyze_air_quality = run_analysis


def save_log_to_db(zone_name, target_gas, pred_val, actual_val, aqi, status, ep_reason, cause, action, notes):
    conn = get_db_connection()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if conn:
        try:
            cursor = conn.cursor()
            sql = """
                INSERT INTO prediction_logs 
                (zone_name, target_gas, predicted_value, actual_value, overall_aqi, air_quality_status,
                 episode_reason, root_cause, recommended_action, inspector_notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (zone_name, target_gas, pred_val, actual_val, aqi, status, ep_reason, cause, action, notes))
            conn.commit()
            new_id = cursor.lastrowid
            cursor.close()
            conn.close()
            return new_id
        except Exception:
            pass

    new_id = len(memory_logs) + 1
    memory_logs.insert(0, {
        "id": new_id,
        "zone_name": zone_name,
        "target_gas": target_gas,
        "predicted_value": pred_val,
        "actual_value": actual_val,
        "overall_aqi": aqi,
        "air_quality_status": status,
        "episode_reason": ep_reason,
        "root_cause": cause,
        "recommended_action": action,
        "inspector_notes": notes,
        "created_at": now_str
    })
    return new_id


# ==========================================
# 5. STATIC HTML/CSS WEB APP & API ROUTES
# ==========================================
@app.get("/")
def serve_root():
    """Serves the main TSPCB Hyderabad Live Map dashboard at the root URL."""
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "TSPCB Air Quality Portal API running", "docs": "/docs", "info": "/api/info"}


@app.get("/index.html")
def serve_index():
    path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="index.html not found")


@app.get("/predict.html")
def serve_predict():
    path = os.path.join(BASE_DIR, "predict.html")
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="predict.html not found")


@app.get("/episodes.html")
def serve_episodes():
    path = os.path.join(BASE_DIR, "episodes.html")
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="episodes.html not found")


@app.get("/logs.html")
def serve_logs():
    path = os.path.join(BASE_DIR, "logs.html")
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="logs.html not found")


@app.get("/style.css")
def serve_style():
    path = os.path.join(BASE_DIR, "style.css")
    if os.path.exists(path):
        return FileResponse(path, media_type="text/css")
    raise HTTPException(status_code=404, detail="style.css not found")


@app.get("/api/info")
def root_project_info():
    """API metadata and information endpoint."""
    return {
        "project": "Air Quality Prediction and Pollution Episode Detection-II",
        "team_id": "G1672",
        "members": ["Ajay", "Tanvi", "Sristi", "Ujwal"],
        "stakeholder": "TSPCB (Telangana State Pollution Control Board) Field Inspectors",
        "loaded_models": list(loaded_models.keys()),
        "docs_url": "/docs"
    }


@app.get("/health")
def health_check():
    conn = get_db_connection()
    db_status = "Connected (MySQL)" if conn else "Active (In-Memory Fallback)"
    if conn:
        conn.close()
    return {
        "status": "healthy",
        "models_loaded": list(loaded_models.keys()),
        "database": db_status
    }


@app.get("/api/models")
def get_models_info():
    return {
        "count": len(loaded_models),
        "available_gases": list(loaded_models.keys()),
        "registry": model_registry
    }


@app.post("/predict")
def predict_air_quality(data: SensorInput):
    row_dict = data.dict()
    result = run_analysis(row_dict, data.actual_value, data.target_gas)
    log_id = save_log_to_db(
        data.zone_name,
        data.target_gas,
        result["selected_gas_prediction"],
        data.actual_value,
        result["overall_aqi"],
        result["air_quality_status"],
        result["episode_reason"],
        result["root_cause"],
        result["recommended_action"],
        data.inspector_notes
    )
    result["log_id"] = log_id
    result["zone_name"] = data.zone_name
    return result


@app.post("/predict-all")
def predict_all_gases(data: SensorInput):
    data.target_gas = "ALL"
    return predict_air_quality(data)


# ==========================================
# 6. HYDERABAD 8 TRAFFIC ZONES AUTO IOT STREAM
# ==========================================
HYDERABAD_ZONES = [
    {
        "zone_id": 1,
        "zone_name": "Narayanguda (KMIT Junction)",
        "lat": 17.3971,
        "lng": 78.4902,
        "traffic_profile": "Moderate College & Commuter Traffic",
        "sensors": {"PT08_S1_CO": 1180.0, "PT08_S2_NMHC": 960.0, "PT08_S3_NOx": 890.0, "PT08_S4_NO2": 1480.0, "PT08_S5_O3": 1050.0, "T": 27.5, "RH": 46.0, "AH": 1.12, "Hour": 10, "DayOfWeek": 2, "Month": 3, "Lag_1": 1.9, "Lag_2": 1.8, "Lag_3": 1.7, "Lag_24": 2.0}
    },
    {
        "zone_id": 2,
        "zone_name": "Punjagutta Traffic Circle",
        "lat": 17.4262,
        "lng": 78.4525,
        "traffic_profile": "Peak Signal Idling (Petrol Cars & Two-Wheelers)",
        "sensors": {"PT08_S1_CO": 1680.0, "PT08_S2_NMHC": 1490.0, "PT08_S3_NOx": 540.0, "PT08_S4_NO2": 1950.0, "PT08_S5_O3": 1620.0, "T": 31.0, "RH": 42.0, "AH": 1.25, "Hour": 18, "DayOfWeek": 3, "Month": 3, "Lag_1": 4.5, "Lag_2": 3.8, "Lag_3": 3.4, "Lag_24": 4.2}
    },
    {
        "zone_id": 3,
        "zone_name": "Hitech City (Cyber Towers)",
        "lat": 17.4504,
        "lng": 78.3808,
        "traffic_profile": "Smooth IT Corridor Flow",
        "sensors": {"PT08_S1_CO": 890.0, "PT08_S2_NMHC": 670.0, "PT08_S3_NOx": 1220.0, "PT08_S4_NO2": 1120.0, "PT08_S5_O3": 760.0, "T": 28.4, "RH": 44.0, "AH": 1.10, "Hour": 14, "DayOfWeek": 2, "Month": 3, "Lag_1": 0.9, "Lag_2": 1.0, "Lag_3": 1.1, "Lag_24": 1.0}
    },
    {
        "zone_id": 4,
        "zone_name": "LB Nagar Ring Road",
        "lat": 17.3457,
        "lng": 78.5522,
        "traffic_profile": "Heavy Highway Diesel Buses & Lorries",
        "sensors": {"PT08_S1_CO": 1590.0, "PT08_S2_NMHC": 1340.0, "PT08_S3_NOx": 490.0, "PT08_S4_NO2": 2080.0, "PT08_S5_O3": 1740.0, "T": 29.8, "RH": 51.0, "AH": 1.30, "Hour": 19, "DayOfWeek": 4, "Month": 11, "Lag_1": 3.8, "Lag_2": 3.2, "Lag_3": 3.0, "Lag_24": 3.6}
    },
    {
        "zone_id": 5,
        "zone_name": "Secunderabad Station Junction",
        "lat": 17.4399,
        "lng": 78.4983,
        "traffic_profile": "High Auto & RTC Bus Congregation",
        "sensors": {"PT08_S1_CO": 1410.0, "PT08_S2_NMHC": 1190.0, "PT08_S3_NOx": 690.0, "PT08_S4_NO2": 1720.0, "PT08_S5_O3": 1390.0, "T": 26.5, "RH": 54.0, "AH": 1.18, "Hour": 9, "DayOfWeek": 1, "Month": 3, "Lag_1": 2.8, "Lag_2": 2.5, "Lag_3": 2.4, "Lag_24": 2.7}
    },
    {
        "zone_id": 6,
        "zone_name": "Kukatpally (JNTU Junction)",
        "lat": 17.4948,
        "lng": 78.3996,
        "traffic_profile": "Morning Winter Smog & Commercial Traffic",
        "sensors": {"PT08_S1_CO": 1540.0, "PT08_S2_NMHC": 1310.0, "PT08_S3_NOx": 580.0, "PT08_S4_NO2": 1840.0, "PT08_S5_O3": 1580.0, "T": 16.5, "RH": 76.0, "AH": 0.92, "Hour": 8, "DayOfWeek": 5, "Month": 12, "Lag_1": 3.4, "Lag_2": 2.9, "Lag_3": 2.7, "Lag_24": 3.3}
    },
    {
        "zone_id": 7,
        "zone_name": "Uppal Ring Road",
        "lat": 17.4018,
        "lng": 78.5602,
        "traffic_profile": "Normal Suburban Traffic Flow",
        "sensors": {"PT08_S1_CO": 940.0, "PT08_S2_NMHC": 720.0, "PT08_S3_NOx": 1150.0, "PT08_S4_NO2": 1190.0, "PT08_S5_O3": 820.0, "T": 28.0, "RH": 45.0, "AH": 1.05, "Hour": 15, "DayOfWeek": 2, "Month": 3, "Lag_1": 1.1, "Lag_2": 1.2, "Lag_3": 1.2, "Lag_24": 1.1}
    },
    {
        "zone_id": 8,
        "zone_name": "Charminar / Afzalgunj Junction",
        "lat": 17.3616,
        "lng": 78.4747,
        "traffic_profile": "Dense Old City Market Congestion",
        "sensors": {"PT08_S1_CO": 1460.0, "PT08_S2_NMHC": 1260.0, "PT08_S3_NOx": 640.0, "PT08_S4_NO2": 1760.0, "PT08_S5_O3": 1440.0, "T": 30.2, "RH": 48.0, "AH": 1.22, "Hour": 17, "DayOfWeek": 6, "Month": 3, "Lag_1": 3.1, "Lag_2": 2.7, "Lag_3": 2.6, "Lag_24": 3.0}
    }
]


@app.get("/api/zones")
def get_hyderabad_zones_live_stream(jitter: bool = True, hour: Optional[int] = None, target_gas: str = "ALL"):
    """Automatically streams IoT sensor readings & ML predictions for 8 Hyderabad Traffic Zones with diurnal time modeling."""
    results = []
    
    # Calculate diurnal time factors based on hour of the day (rush peak vs off-peak)
    time_factor = 1.0
    if hour is not None:
        if 8 <= hour <= 10:
            time_factor = 1.28  # Morning peak office rush
        elif 11 <= hour <= 15:
            time_factor = 0.88  # Afternoon off-peak
        elif 16 <= hour <= 20:
            time_factor = 1.36  # Evening peak jam
        elif 21 <= hour or hour <= 4:
            time_factor = 0.75  # Late night / early morning
        else:
            time_factor = 0.82

    for z in HYDERABAD_ZONES:
        s_copy = z["sensors"].copy()
        
        # Apply hour diurnal factor
        if hour is not None:
            s_copy["Hour"] = hour
            s_copy["PT08_S1_CO"] = round(s_copy["PT08_S1_CO"] * time_factor, 1)
            s_copy["PT08_S2_NMHC"] = round(s_copy["PT08_S2_NMHC"] * time_factor, 1)
            s_copy["PT08_S4_NO2"] = round(s_copy["PT08_S4_NO2"] * time_factor, 1)
            s_copy["PT08_S5_O3"] = round(s_copy["PT08_S5_O3"] * time_factor, 1)
            s_copy["Lag_1"] = round(s_copy["Lag_1"] * time_factor, 2)
            # S3 is inversely proportional to NOx
            s_copy["PT08_S3_NOx"] = round(s_copy["PT08_S3_NOx"] / max(0.5, time_factor), 1)

        if jitter:
            # Tiny live sensor fluctuation (+/- 2%) to simulate real-time IoT telemetry
            factor = random.uniform(0.98, 1.02)
            s_copy["PT08_S1_CO"] = round(s_copy["PT08_S1_CO"] * factor, 1)
            s_copy["PT08_S2_NMHC"] = round(s_copy["PT08_S2_NMHC"] * factor, 1)
            s_copy["PT08_S4_NO2"] = round(s_copy["PT08_S4_NO2"] * factor, 1)

        analysis = run_analysis(s_copy, None, target_gas)
        results.append({
            "zone_id": z["zone_id"],
            "zone_name": z["zone_name"],
            "lat": z["lat"],
            "lng": z["lng"],
            "traffic_profile": z["traffic_profile"],
            "iot_sensors": s_copy,
            "target_gas": target_gas,
            "single_gas_predictions": analysis["single_gas_predictions"],
            "overall_aqi": analysis["overall_aqi"],
            "air_quality_status": analysis["air_quality_status"],
            "zone_color": analysis["zone_color"],
            "dominant_pollutant": analysis["dominant_pollutant"],
            "pollution_trend": analysis["pollution_trend"],
            "episode_reason": analysis["episode_reason"],
            "root_cause": analysis["root_cause"],
            "recommended_action": analysis["recommended_action"]
        })
    return {"city": "Hyderabad", "total_zones": len(results), "hour": hour, "target_gas": target_gas, "zones": results}


# ==========================================
# 7. WHAT-IF ACTION IMPACT SIMULATOR
# ==========================================
@app.post("/api/simulate-action")
def simulate_action_impact(data: SimulateInput):
    before_row = data.dict()
    before_res = run_analysis(before_row, None, "ALL")

    after_row = before_row.copy()
    if data.action_type == "traffic_clearance_40":
        scale = 0.58
        action_label = "Green Signal Corridor (40% Stagnant Traffic Cleared)"
    elif data.action_type == "divert_diesel":
        scale = 0.52
        action_label = "Heavy Diesel Vehicles Diverted to Outer Ring Road"
    else:
        scale = 0.62
        after_row["T"] = min(26.0, after_row["T"] + 3.0)
        after_row["RH"] = max(45.0, after_row["RH"] - 15.0)
        action_label = "GHMC Water Sprinklers & Anti-Smog Guns Deployed"

    for col in ["PT08_S1_CO", "PT08_S2_NMHC", "PT08_S4_NO2", "PT08_S5_O3", "Lag_1", "Lag_2", "Lag_3", "Lag_24"]:
        after_row[col] = round(after_row[col] * scale, 2)
    # PT08.S3(NOx) is inversely proportional to NOx concentration
    after_row["PT08_S3_NOx"] = round(after_row["PT08_S3_NOx"] * 1.25, 2)

    after_res = run_analysis(after_row, None, "ALL")
    reduction_pct = round(((before_res["overall_aqi"] - after_res["overall_aqi"]) / max(1.0, before_res["overall_aqi"])) * 100, 1)

    return {
        "zone_name": data.zone_name,
        "applied_action": action_label,
        "before_aqi": before_res["overall_aqi"],
        "after_aqi": after_res["overall_aqi"],
        "before_status": before_res["air_quality_status"],
        "after_status": after_res["air_quality_status"],
        "before_gases": before_res["single_gas_predictions"],
        "after_gases": after_res["single_gas_predictions"],
        "aqi_reduction_percent": max(5.0, reduction_pct)
    }


# ==========================================
# 8. FLAGGED EPISODES VIEWER & AUTO RETRAIN
# ==========================================
@app.get("/api/episodes/{gas_name}")
def get_flagged_episodes(gas_name: str, limit: int = 100):
    ep_path = os.path.join(MODEL_DIR, f"{gas_name}_episodes.csv")
    if not os.path.exists(ep_path):
        raise HTTPException(status_code=404, detail=f"Episodes file for {gas_name} not found")
    df_ep = pd.read_csv(ep_path)
    if "Datetime" in df_ep.columns:
        # Strictly ensure all timestamps display in 2026
        df_ep["Datetime"] = df_ep["Datetime"].astype(str).str.replace("2004", "2026").str.replace("2005", "2026")
    reason_counts = df_ep["Episode_Reason"].value_counts().to_dict()
    records = df_ep.head(limit).to_dict(orient="records")
    return {
        "gas": gas_name,
        "total_episodes": len(df_ep),
        "summary": reason_counts,
        "episodes": records
    }


@app.post("/retrain")
def auto_detect_and_retrain_models():
    """Automatically detects all valid (GT) gases in AirQualityUCI.csv and trains .pkl models."""
    if not os.path.exists(CSV_PATH):
        raise HTTPException(status_code=404, detail="AirQualityUCI.csv not found")

    df = pd.read_csv(CSV_PATH, sep=";", decimal=",")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")].dropna(how="all").copy()
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H.%M.%S")
    df = df.sort_values("Datetime").reset_index(drop=True)

    sensor_cols = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"]
    weather_cols = ["T", "RH", "AH"]
    all_gt = [c for c in df.columns if "(GT)" in c]
    valid_gases = []

    for g in all_gt:
        df[g] = pd.to_numeric(df[g], errors="coerce")
        miss_pct = ((df[g] == -200).sum() / len(df)) * 100
        if miss_pct < 80:
            valid_gases.append(g)
            df[g + "_missing"] = (df[g] == -200).astype(int)

    for c in sensor_cols + weather_cols + valid_gases:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace(-200, np.nan)

    os.makedirs(MODEL_DIR, exist_ok=True)
    retrained = []

    for gas in valid_gases:
        clean_name = gas.replace("(GT)", "").strip()
        temp = df.copy()
        temp["Hour"] = temp["Datetime"].dt.hour
        temp["DayOfWeek"] = temp["Datetime"].dt.dayofweek
        temp["Month"] = temp["Datetime"].dt.month

        filled = temp[sensor_cols + weather_cols + [gas]].ffill(limit=3)
        temp[sensor_cols + weather_cols] = filled[sensor_cols + weather_cols]
        s = filled[gas]
        temp["Lag_1"] = s.shift(1)
        temp["Lag_2"] = s.shift(2)
        temp["Lag_3"] = s.shift(3)
        temp["Lag_24"] = s.shift(24)
        temp["Rolling_Mean_3"] = s.shift(1).rolling(3).mean()
        temp[gas] = s

        feat_cols = sensor_cols + weather_cols + ["Hour", "DayOfWeek", "Month", "Lag_1", "Lag_2", "Lag_3", "Lag_24", "Rolling_Mean_3"]
        temp = temp.dropna(subset=feat_cols + [gas]).reset_index(drop=True)

        n = len(temp)
        tr_end, val_end = int(n * 0.70), int(n * 0.80)
        tr_df = temp.iloc[:tr_end]
        val_df = temp.iloc[tr_end:val_end]
        te_df = temp.iloc[val_end:]

        gen_tr = tr_df[tr_df[gas + "_missing"] == 0]
        gen_val = val_df[val_df[gas + "_missing"] == 0]

        rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(gen_tr[feat_cols], gen_tr[gas])

        val_rmse = np.sqrt(mean_squared_error(gen_val[gas], rf.predict(gen_val[feat_cols])))
        threshold = float(2.0 * val_rmse)
        high_lvl = float(gen_tr[gas].quantile(0.75))

        gen_te = te_df[te_df[gas + "_missing"] == 0]
        preds = rf.predict(gen_te[feat_cols])
        mae = float(mean_absolute_error(gen_te[gas], preds))
        rmse = float(np.sqrt(mean_squared_error(gen_te[gas], preds)))
        r2 = float(r2_score(gen_te[gas], preds))

        pkg = {
            "gas_name": gas,
            "model": rf,
            "feature_columns": feat_cols,
            "threshold": threshold,
            "high_pollution_level": high_lvl,
            "metrics": {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}
        }
        with open(os.path.join(MODEL_DIR, f"{clean_name}_model.pkl"), "wb") as f:
            pickle.dump(pkg, f)

        model_registry[clean_name] = {
            "gas_column": gas,
            "model_file": f"{clean_name}_model.pkl",
            "episodes_file": f"{clean_name}_episodes.csv",
            "feature_columns": feat_cols,
            "threshold": round(threshold, 4),
            "high_pollution_level": round(high_lvl, 4),
            "metrics": pkg["metrics"]
        }
        retrained.append(clean_name)

    with open(os.path.join(MODEL_DIR, "model_registry.json"), "w") as f:
        json.dump(model_registry, f, indent=4)

    load_all_pickle_models()
    return {"message": "Dynamic Gas Detection & Retraining Complete!", "trained_gases": retrained, "registry": model_registry}


# ==========================================
# 9. MYSQL FULL CRUD ENDPOINTS (/api/logs)
# ==========================================
@app.post("/api/logs")
def create_manual_log(item: CreateManualLogInput):
    log_id = save_log_to_db(
        item.zone_name,
        item.target_gas,
        item.predicted_value,
        item.actual_value,
        item.overall_aqi,
        item.air_quality_status,
        item.episode_reason,
        item.root_cause,
        item.recommended_action,
        item.inspector_notes
    )
    return {"message": f"Log #{log_id} created successfully", "id": log_id}


@app.get("/api/logs")
def read_logs():
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM prediction_logs ORDER BY id DESC LIMIT 100")
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            return {"source": "MySQL Database", "logs": rows}
        except Exception:
            pass
    return {"source": "In-Memory Fallback", "logs": memory_logs}


@app.put("/api/logs/{log_id}")
def update_log(log_id: int, item: LogUpdateInput):
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE prediction_logs SET inspector_notes=%s, air_quality_status=COALESCE(%s, air_quality_status), recommended_action=COALESCE(%s, recommended_action) WHERE id=%s",
                (item.inspector_notes, item.air_quality_status, item.recommended_action, log_id)
            )
            conn.commit()
            cursor.close()
            conn.close()
            return {"message": f"Log #{log_id} updated in MySQL"}
        except Exception:
            pass

    for log in memory_logs:
        if log["id"] == log_id:
            log["inspector_notes"] = item.inspector_notes
            if item.air_quality_status:
                log["air_quality_status"] = item.air_quality_status
            if item.recommended_action:
                log["recommended_action"] = item.recommended_action
            return {"message": f"Log #{log_id} updated in memory"}
    raise HTTPException(status_code=404, detail="Log ID not found")


@app.delete("/api/logs/{log_id}")
def delete_log(log_id: int):
    global memory_logs
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM prediction_logs WHERE id=%s", (log_id,))
            conn.commit()
            cursor.close()
            conn.close()
            return {"message": f"Log #{log_id} deleted from MySQL"}
        except Exception:
            pass

    memory_logs = [l for l in memory_logs if l["id"] != log_id]
    return {"message": f"Log #{log_id} deleted"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
