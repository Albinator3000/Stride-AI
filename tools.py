import os
import json
import math
import time
import random
import datetime as dt
import requests
import folium
from db import save_route, save_run, log_event

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
ORS_URL = "https://api.openrouteservice.org/v2/directions/foot-walking/geojson"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

_PREF_TAGS = {
    "park":      [("leisure", "park"), ("leisure", "garden"), ("landuse", "grass")],
    "river":     [("waterway", "river"), ("waterway", "stream")],
    "water":     [("natural", "water"), ("waterway", "river")],
    "lake":      [("natural", "water")],
    "trail":     [("highway", "path"), ("highway", "footway")],
    "scenic":    [("leisure", "park"), ("natural", "wood"), ("natural", "water")],
    "riverside": [("waterway", "river"), ("waterway", "stream")],
    "green":     [("leisure", "park"), ("natural", "wood"), ("landuse", "forest")],
    "forest":    [("natural", "wood"), ("landuse", "forest")],
    "quiet":     [("highway", "path"), ("leisure", "park")],
    "hilly":     [("natural", "peak"), ("natural", "hill")],
}
_DEFAULT_PREFS = ["park", "scenic"]

_WATER_CLASSES = {"waterway", "natural", "place"}
_WATER_TYPES = {
    "water", "river", "stream", "canal", "bay", "strait",
    "sea", "ocean", "lake", "reservoir", "wetland", "coastline",
    "harbour", "fjord", "spring",
}

_VENUE_SUFFIXES = [
    "apartments", "apartment", "apts", "apt",
    "mall", "shopping center", "shopping centre", "plaza", "centre", "center",
    "complex", "towers", "tower", "suites", "suite", "hotel", "inn",
]


def geocode_location(address: str) -> dict:
    """
    Convert a human-readable address to lat/lon using Nominatim (OSM).
    Falls back to progressively broader queries if the exact name isn't found.
    """
    candidates = _geocode_candidates(address)
    for query in candidates:
        try:
            resp = requests.get(
                NOMINATIM_URL,
                params={"q": query, "format": "json", "limit": 1},
                headers={"User-Agent": "stride-ai-running-agent/1.0"},
                timeout=10,
            )
            resp.raise_for_status()
            results = resp.json()
            if results:
                r = results[0]
                note = f" (searched as '{query}')" if query != address else ""
                return {
                    "status": "success",
                    "lat": float(r["lat"]),
                    "lon": float(r["lon"]),
                    "display_name": r["display_name"] + note,
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return {
        "status": "error",
        "message": (
            f"No location found for '{address}'. "
            "Try a nearby street name, intersection, or landmark instead."
        ),
    }


def _geocode_candidates(address: str) -> list:
    candidates = [address]
    lower = address.lower()
    for suffix in _VENUE_SUFFIXES:
        if lower.endswith(suffix):
            stripped = address[: -len(suffix)].strip().rstrip(",")
            if stripped:
                candidates.append(stripped)
            break
    words = address.split()
    if len(words) > 2 and words[-1].lower() in _VENUE_SUFFIXES:
        candidates.append(" ".join(words[:-1]))
    return candidates


# ---------------------------------------------------------------------------
# Waypoint-based route planning
# ---------------------------------------------------------------------------

# Shape templates: bearing offsets (degrees) for intermediate waypoints.
# seed selects shape + rotation so alternate routes explore different geography.
_SHAPES = [
    [0, 120, 240],            # triangle  — general purpose loop
    [0, 90, 180, 270],        # square    — grid-city friendly
    [0, 72, 144, 216, 288],   # pentagon  — rounder, avoids street reuse
    [0, 90, 180, 90],         # lollipop  — out-and-back with loop at the far end
]


def _bearing_point(lat: float, lon: float, distance_km: float, bearing_deg: float):
    """Return (lat, lon) at `distance_km` and `bearing_deg` from the origin."""
    R = 6371.0
    d = distance_km / R
    b = math.radians(bearing_deg)
    lat1, lon1 = math.radians(lat), math.radians(lon)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(b)
    )
    lon2 = lon1 + math.atan2(
        math.sin(b) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def _is_water_point(lat: float, lon: float) -> bool:
    """Return True if the coordinate is on water (river, sea, lake, etc.)."""
    try:
        resp = requests.get(
            NOMINATIM_REVERSE_URL,
            params={"lat": lat, "lon": lon, "format": "json"},
            headers={"User-Agent": "stride-ai-running-agent/1.0"},
            timeout=5,
        )
        if resp.status_code != 200:
            return False
        data = resp.json()
        return data.get("class", "") in _WATER_CLASSES and data.get("type", "") in _WATER_TYPES
    except Exception:
        return False


# seed 1–8 map to compass bearings N, NE, E, SE, S, SW, W, NW
_SEED_BEARINGS = [0, 45, 90, 135, 180, 225, 270, 315]


def _find_land_bearing(lat: float, lon: float, d_km: float, base_bearing: int) -> int:
    """
    Try bearings in 45° steps until the destination point is on land.
    Returns the first land bearing, or base_bearing if none found.
    """
    for step in range(8):
        b = (base_bearing + step * 45) % 360
        dlat, dlon = _bearing_point(lat, lon, d_km, b)
        time.sleep(0.12)
        if not _is_water_point(dlat, dlon):
            if step > 0:
                print(f"[land] rotated bearing +{step * 45}° to avoid water")
            return b
    print("[land] no land bearing found, using base")
    return base_bearing


# ---------------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------------

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _compass_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Bearing in degrees (0=N, 90=E) from point 1 to point 2."""
    dlon = math.radians(lon2 - lon1)
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


# ---------------------------------------------------------------------------
# Overpass API waypoint selection
# ---------------------------------------------------------------------------

def _query_overpass_waypoints(lat: float, lon: float, radius_m: float, prefs: list) -> list:
    """
    Query Overpass for named OSM features (parks, water, paths) near a point.
    Returns list of {"lat", "lon", "name", "type"} dicts.
    """
    seen_tags: set = set()
    tag_filters = []
    for pref in prefs:
        for k, v in _PREF_TAGS.get(pref, []):
            if (k, v) not in seen_tags:
                seen_tags.add((k, v))
                tag_filters.append((k, v))

    if not tag_filters:
        tag_filters = [("leisure", "park"), ("natural", "water"), ("natural", "wood")]

    r = int(radius_m)
    parts = []
    for k, v in tag_filters[:8]:
        parts.append(f'node["{k}"="{v}"](around:{r},{lat},{lon});')
        parts.append(f'way["{k}"="{v}"](around:{r},{lat},{lon});')

    query = "[out:json][timeout:15];\n(\n" + "\n".join(parts) + "\n);\nout center;"
    try:
        resp = requests.post(
            OVERPASS_URL,
            data={"data": query},
            headers={"User-Agent": "stride-ai-running-agent/1.0"},
            timeout=25,
        )
        if resp.status_code != 200:
            return []
        results = []
        for el in resp.json().get("elements", []):
            if el["type"] == "node":
                feat_lat, feat_lon = el["lat"], el["lon"]
            elif el["type"] == "way" and "center" in el:
                feat_lat, feat_lon = el["center"]["lat"], el["center"]["lon"]
            else:
                continue
            tags = el.get("tags", {})
            name = tags.get("name", "")
            feat_type = next(
                (tags[k] for k in ("leisure", "natural", "waterway", "landuse") if k in tags),
                "feature",
            )
            results.append({"lat": feat_lat, "lon": feat_lon, "name": name, "type": feat_type})
        return results
    except Exception:
        return []


def _select_loop_waypoints(
    start_lat: float,
    start_lon: float,
    candidates: list,
    n: int = 2,
    target_radius_km: float = 2.0,
) -> list:
    """
    Pick n waypoints from candidates spread across n compass sectors.
    Prefers candidates whose distance from start is near target_radius_km.
    Filters out candidates closer than 0.3 km (routing noise) or with no road access.
    """
    if not candidates:
        return []

    annotated = []
    for c in candidates:
        dist = _haversine(start_lat, start_lon, c["lat"], c["lon"])
        if dist < 0.3:
            continue
        bearing = _compass_bearing(start_lat, start_lon, c["lat"], c["lon"])
        annotated.append({**c, "dist_km": dist, "bearing": bearing,
                          "radius_score": abs(dist - target_radius_km)})

    sector_size = 360.0 / n
    # Random sector rotation so the N/S split varies each call
    rotation = random.uniform(0, sector_size)
    selected = []
    used = set()

    for i in range(n):
        lo = (rotation + i * sector_size) % 360
        hi = (lo + sector_size) % 360
        in_sector = [
            c for c in annotated
            if (lo < hi and lo <= c["bearing"] < hi)
            or (lo >= hi and (c["bearing"] >= lo or c["bearing"] < hi))
            and c.get("name", "") not in used
        ]
        if not in_sector:
            in_sector = [c for c in annotated if c.get("name", "") not in used]
        if not in_sector:
            break
        # Pick randomly from the top-3 by radius proximity for variety
        pool = sorted(in_sector, key=lambda c: c["radius_score"])[:3]
        best = random.choice(pool)
        selected.append(best)
        if best.get("name"):
            used.add(best["name"])

    return selected


def _plan_waypoint_route(
    lat: float,
    lon: float,
    distance_km: float,
    start_address: str,
    ors_request,
    waypoints: list,
) -> dict | None:
    """
    Route start → waypoints → start using ORS multi-point directions.
    Bisects by scaling waypoint orbit radius to converge on target distance.
    Returns None if ORS rejects the waypoints (caller falls back to round_trip).
    """
    TOLERANCE_KM = 0.4
    MAX_ITER = 6

    def _build_coords(scale: float) -> list:
        coords = [[lon, lat]]
        for wp in waypoints:
            wp_lat = lat + (wp["lat"] - lat) * scale
            wp_lon = lon + (wp["lon"] - lon) * scale
            coords.append([wp_lon, wp_lat])
        coords.append([lon, lat])
        return coords

    best_route = None
    best_delta = float("inf")
    best_in_window = False
    scale = 1.0
    low_scale: float | None = None
    high_scale: float | None = None

    for it in range(MAX_ITER):
        payload = {
            "coordinates": _build_coords(scale),
            "instructions": True,
            "elevation": True,
            "options": {"avoid_features": ["ferries"]},
        }
        resp = ors_request(payload)
        if resp.status_code != 200:
            print(f"[route] waypoint ORS error {resp.status_code}: {resp.text[:200]}")
            return None

        feature = _first_feature(resp)
        if not feature:
            return None

        actual_km = feature["properties"]["summary"]["distance"] / 1000
        delta_km = actual_km - distance_km
        print(
            f"[route] waypoint it={it+1} scale={scale:.3f} "
            f"→ actual={actual_km:.2f}km (target {distance_km}km)"
        )

        if 0 <= delta_km <= TOLERANCE_KM:
            best_route = feature
            best_in_window = True
            break

        if abs(delta_km) < best_delta:
            best_delta = abs(delta_km)
            best_route = feature

        if delta_km > TOLERANCE_KM:   # too long — bring waypoints closer
            high_scale = scale
            scale = (low_scale + high_scale) / 2 if low_scale is not None else scale * distance_km / actual_km
        else:                          # too short — push waypoints further out
            low_scale = scale
            scale = (low_scale + high_scale) / 2 if high_scale is not None else scale * distance_km / actual_km

        scale = max(0.1, min(scale, 5.0))

    if best_route is None:
        return None

    wp_summary = [{"name": wp.get("name", ""), "type": wp.get("type", "")} for wp in waypoints]
    return _package_route(
        best_route,
        distance_km=distance_km,
        start_address=start_address,
        route_type="out_and_back",
        in_window=best_in_window,
        extra={"waypoints": wp_summary},
    )


def plan_running_route(
    lat: float,
    lon: float,
    distance_km: float,
    start_address: str,
    seed: int = 1,
    route_type: str = "out_and_back",
    preferences: list = None,
) -> dict:
    """
    Generate a running route in one of two modes:

    out_and_back  — closed loop using real OSM scenic waypoints (parks, rivers,
                    green spaces) when available, falling back to ORS round_trip.

    point_to_point — straight line from start in the seed direction, no return.

    seed selects the compass heading (1=N, 2=NE, … 8=NW).
    preferences is an optional list of terrain keywords: "park", "river", "scenic",
    "trail", "green", "forest", "quiet", "hilly", "riverside", "lake", "water".
    """
    ors_api_key = os.getenv("ORS_API_KEY")
    if not ors_api_key:
        return {"status": "error", "message": "ORS_API_KEY not set in environment"}

    headers = {"Authorization": ors_api_key, "Content-Type": "application/json"}

    def _ors_request(payload):
        time.sleep(0.5)
        for attempt in range(3):
            resp = requests.post(ORS_URL, json=payload, headers=headers, timeout=25)
            if resp.status_code != 429:
                return resp
            time.sleep(2 ** (attempt + 1))
        return resp

    try:
        if route_type == "out_and_back":
            prefs = preferences or _DEFAULT_PREFS
            # Rough orbit radius for a circular loop of target distance
            target_radius_km = distance_km / (2 * math.pi)
            search_radius_m = target_radius_km * 1000 * 1.5

            candidates = _query_overpass_waypoints(lat, lon, search_radius_m, prefs)
            waypoints = _select_loop_waypoints(
                lat, lon, candidates, n=2, target_radius_km=target_radius_km
            )

            # Randomize ORS seed when user hasn't explicitly picked one,
            # so repeat requests produce different loop geometry
            effective_seed = seed if seed != 1 else random.randint(1, 8)

            if waypoints:
                names = [w.get("name") or w.get("type", "?") for w in waypoints]
                print(f"[route] Routing via {len(waypoints)} OSM waypoints: {names}")
                result = _plan_waypoint_route(
                    lat, lon, distance_km, start_address, _ors_request, waypoints
                )
                if result:
                    return result
                print("[route] Waypoint routing failed — falling back to round_trip")
            else:
                print(f"[route] No Overpass waypoints found for prefs={prefs}, using round_trip")

            return _plan_round_trip(
                lat, lon, distance_km, start_address, effective_seed, _ors_request
            )
        return _plan_point_to_point(
            lat, lon, distance_km, start_address, seed, _ors_request
        )
    except requests.HTTPError as e:
        return {"status": "error", "message": f"ORS API error: {e.response.text}"}
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _plan_round_trip(lat, lon, distance_km, start_address, seed, ors_request) -> dict:
    """
    Generate a closed-loop running route using ORS's round_trip algorithm.

    Strategy: ORS's `length` parameter doesn't reliably produce a route of that
    length — actual varies 10–30%. We iterate, scaling `length` based on the
    ratio between requested and actual, and also try different seeds which give
    different loop geometries of similar length.
    """
    TOLERANCE_KM = 0.3
    MAX_ITER = 8
    target_m = distance_km * 1000
    base_bearing = _SEED_BEARINGS[(seed - 1) % len(_SEED_BEARINGS)]

    def _payload(length_m, current_seed):
        return {
            "coordinates": [[lon, lat]],
            "instructions": True,
            "elevation": True,
            "options": {
                "avoid_features": ["ferries"],
                "round_trip": {
                    "length": int(length_m),
                    "points": 8,
                    "seed": current_seed,
                },
            },
        }

    best_route = None
    best_delta = float("inf")
    best_in_window = False
    tried_seeds = []

    def _try_seed(current_seed, budget):
        """Bisect on `length` for a single seed. Returns (route, delta_km, in_window)."""
        nonlocal best_route, best_delta
        length_m = target_m
        low_length = None   # length that produced actual < target
        high_length = None  # length that produced actual > target + tolerance
        local_best = None
        local_best_delta = float("inf")
        for it in range(budget):
            resp = ors_request(_payload(length_m, current_seed))
            if resp.status_code != 200:
                return local_best, local_best_delta, False

            feature = _first_feature(resp)
            if not feature:
                return local_best, local_best_delta, False

            actual_km = feature["properties"]["summary"]["distance"] / 1000
            delta_km = actual_km - distance_km
            print(
                f"[route] seed={current_seed} it={it+1} length={length_m/1000:.2f}km "
                f"→ actual={actual_km:.2f}km (target {distance_km}km)"
            )

            if 0 <= delta_km <= TOLERANCE_KM:
                if abs(delta_km) < best_delta:
                    best_delta = abs(delta_km)
                    best_route = feature
                return feature, delta_km, True

            if abs(delta_km) < local_best_delta:
                local_best_delta = abs(delta_km)
                local_best = feature
            if abs(delta_km) < best_delta:
                best_delta = abs(delta_km)
                best_route = feature

            # Bisect on `length` when we've seen both an over and an under
            if delta_km > TOLERANCE_KM:
                high_length = length_m
                if low_length is not None:
                    length_m = (low_length + high_length) / 2
                else:
                    length_m *= target_m / (actual_km * 1000)
            else:
                low_length = length_m
                if high_length is not None:
                    length_m = (low_length + high_length) / 2
                else:
                    length_m *= target_m / (actual_km * 1000)
            length_m = max(target_m * 0.3, min(length_m, target_m * 3.0))

        return local_best, local_best_delta, False

    # Try up to 4 seeds, bisecting on each. Bail early on the first in-window hit.
    remaining = MAX_ITER
    for s_offset in range(4):
        if remaining <= 0:
            break
        current_seed = seed + s_offset
        tried_seeds.append(current_seed)
        budget = min(4, remaining)
        _route, _delta, in_window = _try_seed(current_seed, budget)
        remaining -= budget
        if in_window:
            best_in_window = True
            break

    if best_route is None:
        raise ValueError("Could not generate a valid loop for this location.")

    return _package_route(
        best_route,
        distance_km=distance_km,
        start_address=start_address,
        route_type="out_and_back",
        in_window=best_in_window,
        extra={"bearing": base_bearing, "seeds_tried": tried_seeds},
    )


def _plan_point_to_point(lat, lon, distance_km, start_address, seed, ors_request) -> dict:
    """One-way route in a chosen compass direction, distance_km total."""
    TOLERANCE_KM = 0.3
    MAX_ITER = 6
    base_bearing = _SEED_BEARINGS[(seed - 1) % len(_SEED_BEARINGS)]
    bearing = _find_land_bearing(lat, lon, distance_km, base_bearing)

    d_km = distance_km
    best_route = None
    best_delta = float("inf")
    best_in_window = False

    for iteration in range(MAX_ITER):
        dlat, dlon = _bearing_point(lat, lon, d_km, bearing)
        payload = {
            "coordinates": [[lon, lat], [dlon, dlat]],
            "instructions": True,
            "elevation": True,
            "options": {"avoid_features": ["ferries"]},
        }
        resp = ors_request(payload)
        if resp.status_code != 200:
            break

        feature = _first_feature(resp)
        if not feature:
            break

        actual_km = feature["properties"]["summary"]["distance"] / 1000
        delta_km = actual_km - distance_km
        print(
            f"[route] iter={iteration+1} p2p bearing={bearing}° d={d_km:.2f}km "
            f"→ actual={actual_km:.2f}km (target {distance_km}km)"
        )

        if 0 <= delta_km <= TOLERANCE_KM:
            best_route = feature
            best_in_window = True
            break

        if abs(delta_km) < best_delta:
            best_delta = abs(delta_km)
            best_route = feature

        d_km *= distance_km / actual_km

    if best_route is None:
        raise ValueError("Could not generate a point-to-point route for this location.")

    return _package_route(
        best_route,
        distance_km=distance_km,
        start_address=start_address,
        route_type="point_to_point",
        in_window=best_in_window,
        extra={"bearing": bearing},
    )


def _first_feature(resp):
    """Extract the first feature from an ORS GeoJSON response, or None."""
    try:
        features = resp.json().get("features", [])
    except Exception:
        return None
    return features[0] if features else None


def _package_route(feature, distance_km, start_address, route_type, in_window, extra=None) -> dict:
    props = feature["properties"]
    all_steps = [s for seg in props.get("segments", []) for s in seg.get("steps", [])]
    actual_km = round(props["summary"]["distance"] / 1000, 2)
    delta_km = round(actual_km - distance_km, 2)
    # GeoJSON coords are [[lon, lat, ele?], ...]; strip elevation if present
    coords = [[c[0], c[1]] for c in feature["geometry"]["coordinates"]]
    steps = _format_steps(all_steps)
    route_id = save_route(coords)
    map_path, osm_url = _create_map(coords, start_address)

    result = {
        "status": "success",
        "route_id": route_id,
        "route_type": route_type,
        "total_distance_km": actual_km,
        "elevation_gain_m": round(props.get("ascent", props["summary"].get("ascent", 0)), 1),
        "steps": steps,
        "map_path": map_path,
        "osm_url": osm_url,
    }
    if extra:
        result.update(extra)

    if not in_window:
        if delta_km < 0:
            result["distance_warning"] = (
                f"Best loop found is {actual_km}km — {abs(delta_km)}km short of the "
                f"target {distance_km}km. Try a different seed or starting point."
            )
        elif delta_km > 0.3:
            result["distance_warning"] = (
                f"Closest loop is {actual_km}km — {delta_km}km over target. "
                f"Try a different seed for a tighter fit."
            )

    return result


# ---------------------------------------------------------------------------
# Direction formatting
# ---------------------------------------------------------------------------

def _format_steps(raw_steps: list) -> list:
    """
    Clean ORS step instructions into human-readable running directions.
    - Strips ORS unnamed-road placeholders ("-", "--")
    - Drops very short unnamed segments (routing noise)
    - Merges consecutive steps on the same named street
    - Formats distances as m or km
    """
    CARDINAL_ONLY = {
        "Head north", "Head south", "Head east", "Head west",
        "Head northeast", "Head northwest", "Head southeast", "Head southwest",
    }
    MIN_UNNAMED_M = 80
    MIN_NAMED_M = 30
    _UNNAMED = {"-", "--", "none", "unknown", ""}

    import re as _re

    def _clean_name(raw):
        return "" if (raw or "").strip() in _UNNAMED else (raw or "").strip()

    def _strip_ors_unnamed(text):
        return _re.sub(r"\s+on\s+(-{1,2})\s*$", "", text).strip()

    cleaned = []
    for i, step in enumerate(raw_steps):
        instruction = _strip_ors_unnamed(step.get("instruction", "").strip())
        dist_m = step.get("distance", 0)
        name = _clean_name(step.get("name", ""))

        if not instruction:
            continue
        if "Arrive" in instruction and dist_m == 0:
            continue
        if not name and dist_m < MIN_UNNAMED_M:
            continue
        if name and dist_m < MIN_NAMED_M:
            continue

        is_cardinal = any(instruction.startswith(c) for c in CARDINAL_ONLY)
        if is_cardinal:
            if name:
                instruction = f"{instruction} on {name}"
            elif i > 0 and dist_m < 150:
                continue
        elif name and "onto" not in instruction and " on " not in instruction:
            instruction = f"{instruction} on {name}"

        dist_str = f"{dist_m:.0f}m" if dist_m < 1000 else f"{dist_m / 1000:.1f}km"
        cleaned.append(f"{instruction} ({dist_str})")

    merged = []
    for step_str in cleaned:
        if merged and _same_street(merged[-1], step_str):
            merged[-1] = _merge_step_distances(merged[-1], step_str)
        else:
            merged.append(step_str)
    return merged


def _same_street(a: str, b: str) -> bool:
    import re
    ma = re.search(r" on (.+?) \(", a)
    mb = re.search(r" on (.+?) \(", b)
    return bool(ma and mb and ma.group(1) == mb.group(1))


def _merge_step_distances(a: str, b: str) -> str:
    import re

    def extract_m(s):
        m = re.search(r"\((\d+(?:\.\d+)?)(m|km)\)$", s)
        return float(m.group(1)) * (1000 if m.group(2) == "km" else 1) if m else 0

    total_m = extract_m(a) + extract_m(b)
    dist_str = f"{total_m:.0f}m" if total_m < 1000 else f"{total_m / 1000:.1f}km"
    return re.sub(r"\([^)]+\)$", f"({dist_str})", a)


# ---------------------------------------------------------------------------
# Map rendering
# ---------------------------------------------------------------------------

def _create_map(coordinates: list, start_address: str, output_dir: str = "runs") -> tuple:
    """Render route as a Folium HTML map. Returns (file_path, osm_url)."""
    os.makedirs(output_dir, exist_ok=True)
    latlon = [[c[1], c[0]] for c in coordinates]
    start = latlon[0]

    m = folium.Map(location=start, zoom_start=14, tiles="OpenStreetMap")
    folium.PolyLine(latlon, color="#E85D04", weight=4, opacity=0.85).add_to(m)
    folium.Marker(
        location=start,
        popup=f"Start / Finish: {start_address}",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{timestamp}_map.html"
    m.save(filename)

    lats = [c[0] for c in latlon]
    lons = [c[1] for c in latlon]
    bbox = f"{min(lons)},{min(lats)},{max(lons)},{max(lats)}"
    osm_url = f"https://www.openstreetmap.org/?bbox={bbox}&layer=mapnik"
    return filename, osm_url


# ---------------------------------------------------------------------------
# Polyline decoding
# ---------------------------------------------------------------------------

def _decode_polyline(encoded: str) -> list:
    """Decode a Google-encoded polyline string into [[lon, lat], ...] pairs."""
    coords = []
    index = 0
    lat = 0
    lng = 0
    while index < len(encoded):
        pair = {}
        for is_lng in (False, True):
            shift, result = 0, 0
            while index < len(encoded):
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            else:
                break  # exhausted mid-coordinate — discard partial pair
            pair[is_lng] = ~(result >> 1) if result & 1 else result >> 1
        if len(pair) == 2:
            lat += pair[False]
            lng += pair[True]
            coords.append([lng / 1e5, lat / 1e5])
    return coords


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def export_route_gpx(route_id: str, title: str = "Running Route", output_dir: str = "runs") -> dict:
    """Export a saved route as a GPX file compatible with Garmin, Wahoo, and any GPS device."""
    from db import get_route
    coords = get_route(route_id)
    if coords is None:
        return {"status": "error", "message": f"No route found for id '{route_id}'."}

    os.makedirs(output_dir, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{timestamp}_route.gpx"

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="Stride AI"',
        '  xmlns="http://www.topografix.com/GPX/1/1"',
        '  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '  xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd">',
        f'  <metadata><name>{title}</name></metadata>',
        "  <trk>",
        f"    <name>{title}</name>",
        "    <trkseg>",
    ]
    for lon, lat in coords:
        lines.append(f'      <trkpt lat="{lat}" lon="{lon}"></trkpt>')
    lines += ["    </trkseg>", "  </trk>", "</gpx>"]

    with open(filename, "w") as f:
        f.write("\n".join(lines))

    return {"status": "success", "gpx_path": filename, "message": f"GPX saved to {filename}"}


def save_run_plan(
    title: str,
    location: str,
    distance_km: float,
    total_distance_km: float,
    elevation_gain_m: float,
    steps: list,
    map_path: str,
    route_id: str = None,
    output_dir: str = "runs",
) -> dict:
    """Persist the run plan to JSON and record it in the analytics DB."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    plan = {
        "title": title,
        "start_location": location,
        "requested_distance_km": distance_km,
        "actual_distance_km": total_distance_km,
        "elevation_gain_m": elevation_gain_m,
        "directions": steps,
        "map_path": map_path,
        "route_id": route_id,
        "created_at": dt.datetime.now().isoformat(),
    }
    filename = f"{output_dir}/{timestamp}_plan.json"
    with open(filename, "w") as f:
        json.dump(plan, f, indent=2)

    save_run(
        title=title,
        start_location=location,
        requested_km=distance_km,
        actual_km=total_distance_km,
        elevation_m=elevation_gain_m,
        route_id=route_id,
        map_path=map_path,
    )

    return {
        "status": "success",
        "plan_path": filename,
        "message": f"Run plan '{title}' saved.",
    }
