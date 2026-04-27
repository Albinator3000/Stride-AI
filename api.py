import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tools import geocode_location, plan_running_route, export_route_gpx
from db import save_run, get_run_history, get_route, get_analytics, get_recent_route_coords
from agent import stream_agent

app = FastAPI(title="Stride AI", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

import math


def _novelty_score(new_coords: list, past_runs: list[list], threshold_m: float = 40.0) -> int:
    """
    Return 0–100 indicating how much of the new route covers ground not run before.
    100 = entirely new. 0 = completely retracing past routes.
    Samples every 5th coordinate for speed; threshold_m is the proximity radius.
    """
    if not past_runs or not new_coords:
        return 100

    # Build a flat list of (lat, lon) from all past runs, sampled every 5 points
    past_pts: list[tuple[float, float]] = []
    for coords in past_runs:
        for c in coords[::5]:
            past_pts.append((c[1], c[0]))  # coords stored as [lon, lat]

    if not past_pts:
        return 100

    # Convert threshold to approximate degree delta at the start lat
    start_lat = new_coords[0][1]
    lat_deg = threshold_m / 111_000
    lon_deg = threshold_m / (111_000 * math.cos(math.radians(start_lat)))

    new_sample = new_coords[::5]
    novel = 0
    for c in new_sample:
        lat, lon = c[1], c[0]
        seen = any(
            abs(lat - p[0]) <= lat_deg and abs(lon - p[1]) <= lon_deg
            for p in past_pts
        )
        if not seen:
            novel += 1

    return round(novel / len(new_sample) * 100)


class RouteRequest(BaseModel):
    address: str
    distance_km: float = Field(..., gt=0, le=100)
    seed: int = Field(1, ge=1, le=8)
    route_type: str = Field("out_and_back", pattern="^(out_and_back|point_to_point)$")


class RouteResponse(BaseModel):
    route_id: str
    address: str
    route_type: str
    distance_km: float
    elevation_gain_m: float
    novelty_score: int
    coordinates: list[list[float]]
    steps: list[str]
    distance_warning: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/routes/generate", response_model=RouteResponse)
def generate_route(req: RouteRequest):
    geo = geocode_location(req.address)
    if geo["status"] == "error":
        raise HTTPException(status_code=422, detail=geo["message"])

    route = plan_running_route(
        lat=geo["lat"],
        lon=geo["lon"],
        distance_km=req.distance_km,
        start_address=geo["display_name"],
        seed=req.seed,
        route_type=req.route_type,
    )
    if route["status"] == "error":
        raise HTTPException(status_code=422, detail=route["message"])

    coords = get_route(route["route_id"])
    past = get_recent_route_coords(limit=20)
    novelty = _novelty_score(coords, past)

    save_run(
        title=f"{req.address} {req.distance_km}km",
        start_location=req.address,
        requested_km=req.distance_km,
        actual_km=route["total_distance_km"],
        elevation_m=route["elevation_gain_m"],
        route_id=route["route_id"],
    )

    return RouteResponse(
        route_id=route["route_id"],
        address=geo["display_name"],
        route_type=route["route_type"],
        distance_km=route["total_distance_km"],
        elevation_gain_m=route["elevation_gain_m"],
        novelty_score=novelty,
        coordinates=coords,
        steps=route["steps"],
        distance_warning=route.get("distance_warning"),
    )


class StreamRequest(BaseModel):
    message: str
    history: list[dict] | None = None


@app.post("/routes/stream")
def stream_route(req: StreamRequest):
    return StreamingResponse(
        stream_agent(req.message, req.history),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/routes/{route_id}/gpx")
def download_gpx(route_id: str, title: str = Query("Running Route")):
    result = export_route_gpx(route_id=route_id, title=title)
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    path = result["gpx_path"]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="GPX file not found.")
    return FileResponse(path, media_type="application/gpx+xml", filename=os.path.basename(path))


# ---------------------------------------------------------------------------
# Run history
# ---------------------------------------------------------------------------

@app.get("/runs")
def list_runs(limit: int = Query(20, ge=1, le=100), location: str | None = None):
    return get_run_history(limit=limit, location=location)


@app.get("/runs/analytics")
def runs_analytics():
    return get_analytics()


@app.get("/routes/{route_id}")
def get_route_coords(route_id: str):
    coords = get_route(route_id)
    if coords is None:
        raise HTTPException(status_code=404, detail="Route not found.")
    return {"route_id": route_id, "coordinates": coords}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return FileResponse("static/index.html")
