"""
FastAPI application for AI-Powered Air Quality Prediction & Decision Support System (v2.0).
"""

import os
from pathlib import Path
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from backend.schemas import (
    PredictRequest,
    PredictResponse,
    ModelInfoResponse,
    HealthResponse,
    ExampleResponse
)
from backend.predictor import (
    load_pipeline,
    load_metadata,
    is_model_ready,
    get_model_info_dict,
    predict_aqi_composite,
    get_training_maxima
)
from backend.decision_support import (
    classify_aqi,
    get_recommendations,
    check_training_maxima_warnings
)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Pre-loads the serialized model and metadata once at startup.
    """
    try:
        load_pipeline()
        load_metadata()
        print("[Startup] AirSense AI Model v2.0 and metadata loaded successfully.")
    except Exception as e:
        print(f"[Startup Warning] Model artifact could not be preloaded: {e}")
    yield


app = FastAPI(
    title="AI-Powered Air Quality Prediction & Decision Support API",
    description=(
        "FastAPI backend for predicting next-calendar-day AQI from current-day pollutant "
        "readings and providing rule-based environmental decision support with physical particulate imputation."
    ),
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for frontend assets
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    """Custom validation error handler with clean field-level error messages."""
    errors = []
    for err in exc.errors():
        field_path = " -> ".join(str(loc) for loc in err.get("loc", []))
        msg = err.get("msg", "Validation error")
        errors.append(f"{field_path}: {msg}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error", "errors": errors}
    )


@app.get("/", include_in_schema=False)
async def serve_root():
    """Serves the dashboard placeholder HTML page at root URL."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend placeholder index.html not found."
        )
    return FileResponse(str(index_file))


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint reporting API operational readiness
    and model deserialization status.
    """
    ready = is_model_ready()
    if not ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "unhealthy", "model_loaded": False, "message": "Model is not loaded."}
        )
    return HealthResponse(status="ok", model_loaded=True, model_version="2.0.0")


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Returns architecture, genuine feature importances from the trained estimator,
    training maxima, and reported validation metrics.
    """
    try:
        info = get_model_info_dict()
        return ModelInfoResponse(**info)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model information: {str(e)}"
        )


@app.get("/example", response_model=ExampleResponse, tags=["Model"])
async def get_test_example():
    """
    Returns the documented historical Visakhapatnam test example (2020-06-22)
    from the Stage 1 training report.
    """
    return ExampleResponse(
        city="Visakhapatnam",
        date="2020-06-22",
        input_values={
            "pm25": 33.17,
            "pm10": 108.22,
            "no2": 42.45,
            "co": 0.73,
            "so2": 13.65,
            "o3": 34.85
        },
        today_aqi=95.0,
        actual_next_day_aqi=100.0,
        reported_model_prediction=112.63,
        reported_absolute_error=12.63,
        note="Historical reference test sample from model evaluation. Not live sensor readings."
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_next_day_aqi_endpoint(payload: PredictRequest):
    """
    Predicts next-calendar-day AQI from current-day pollutant readings,
    assigns an operational category, checks training bounds, and generates recommendations.
    Applies monthly particulate ratio estimation when PM10 or PM2.5 is omitted.
    """
    try:
        # 1. Execute composite inference (handles ratio imputation, CO threshold, bounds clamping)
        res = predict_aqi_composite(
            pm25=payload.pm25,
            pm10=payload.pm10,
            no2=payload.no2,
            co=payload.co,
            so2=payload.so2,
            o3=payload.o3,
            month=payload.month
        )

        # 2. Check for inputs exceeding observed training partition maxima
        training_maxima = get_training_maxima()
        # Only check provided non-null inputs
        provided_inputs = {k: v for k, v in payload.to_feature_dict().items() if v is not None}
        maxima_warnings = check_training_maxima_warnings(provided_inputs, training_maxima)

        # 3. Collate all warnings
        all_warnings = res.get('warnings', []) + maxima_warnings

        # 4. Retrieve rule-based environmental decision support recommendations
        recommendations = get_recommendations(res['aqi_category'])

        return PredictResponse(
            predicted_aqi=res['predicted_aqi'],
            raw_predicted_aqi=res['raw_predicted_aqi'],
            category=res['aqi_category'],
            recommendations=recommendations,
            warnings=all_warnings,
            input_values=payload.model_dump(),
            features_used=res.get('features_used'),
            model_name="AirSense AI Release Candidate v2.0 (RandomForestRegressor)",
            model_version=res.get('model_version', "2.0.0")
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference processing failed: {str(e)}"
        )
