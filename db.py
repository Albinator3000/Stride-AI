import sqlite3
import json
import uuid
import os
import datetime as dt

DB_PATH = os.getenv("DB_PATH", "runs/routes.db")


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS routes (
                id          TEXT PRIMARY KEY,
                coordinates TEXT NOT NULL,
                created_at  TEXT NOT NULL
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id              TEXT PRIMARY KEY,
                title           TEXT NOT NULL,
                start_location  TEXT NOT NULL,
                requested_km    REAL NOT NULL,
                actual_km       REAL NOT NULL,
                elevation_m     REAL NOT NULL,
                route_id        TEXT,
                map_path        TEXT,
                created_at      TEXT NOT NULL
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id          TEXT PRIMARY KEY,
                event_type  TEXT NOT NULL,
                run_id      TEXT,
                payload     TEXT,
                created_at  TEXT NOT NULL
            )
        """)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def save_route(coordinates: list) -> str:
    init_db()
    route_id = str(uuid.uuid4())
    with _conn() as con:
        con.execute(
            "INSERT INTO routes (id, coordinates, created_at) VALUES (?, ?, ?)",
            (route_id, json.dumps(coordinates), dt.datetime.now().isoformat()),
        )
    return route_id


def get_recent_route_coords(limit: int = 20) -> list[list]:
    """Return coordinates for the most recent `limit` saved runs (for novelty scoring)."""
    init_db()
    with _conn() as con:
        rows = con.execute(
            """SELECT r.coordinates FROM routes r
               JOIN runs u ON u.route_id = r.id
               ORDER BY u.created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [json.loads(row[0]) for row in rows]


def get_route(route_id: str) -> list | None:
    init_db()
    with _conn() as con:
        row = con.execute(
            "SELECT coordinates FROM routes WHERE id = ?", (route_id,)
        ).fetchone()
    return json.loads(row[0]) if row else None


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def save_run(
    title: str,
    start_location: str,
    requested_km: float,
    actual_km: float,
    elevation_m: float,
    route_id: str | None = None,
    map_path: str | None = None,
) -> str:
    init_db()
    run_id = str(uuid.uuid4())
    with _conn() as con:
        con.execute(
            """INSERT INTO runs
               (id, title, start_location, requested_km, actual_km, elevation_m,
                route_id, map_path, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, title, start_location, requested_km, actual_km, elevation_m,
             route_id, map_path, dt.datetime.now().isoformat()),
        )
    log_event("run_saved", run_id=run_id, payload={"title": title, "actual_km": actual_km})
    return run_id


def get_run_history(limit: int = 20, location: str | None = None) -> list[dict]:
    init_db()
    with _conn() as con:
        if location:
            rows = con.execute(
                "SELECT * FROM runs WHERE start_location LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"%{location}%", limit),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def log_event(event_type: str, run_id: str | None = None, payload: dict | None = None):
    init_db()
    with _conn() as con:
        con.execute(
            "INSERT INTO events (id, event_type, run_id, payload, created_at) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), event_type, run_id,
             json.dumps(payload) if payload else None,
             dt.datetime.now().isoformat()),
        )


def get_events(event_type: str | None = None, limit: int = 50) -> list[dict]:
    init_db()
    with _conn() as con:
        if event_type:
            rows = con.execute(
                "SELECT * FROM events WHERE event_type = ? ORDER BY created_at DESC LIMIT ?",
                (event_type, limit),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM events ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def get_analytics() -> dict:
    """Aggregate stats for the UI dashboard."""
    init_db()
    with _conn() as con:
        totals = con.execute("""
            SELECT
                COUNT(*)            AS total_runs,
                COALESCE(SUM(actual_km), 0)    AS total_km,
                COALESCE(AVG(actual_km), 0)    AS avg_km,
                COALESCE(SUM(elevation_m), 0)  AS total_elevation_m,
                COALESCE(AVG(elevation_m), 0)  AS avg_elevation_m
            FROM runs
        """).fetchone()

        top_locations = con.execute("""
            SELECT start_location, COUNT(*) AS count, COALESCE(SUM(actual_km), 0) AS total_km
            FROM runs
            GROUP BY start_location
            ORDER BY count DESC
            LIMIT 5
        """).fetchall()

        by_week = con.execute("""
            SELECT
                strftime('%Y-W%W', created_at) AS week,
                COUNT(*)                        AS runs,
                COALESCE(SUM(actual_km), 0)    AS total_km
            FROM runs
            GROUP BY week
            ORDER BY week DESC
            LIMIT 8
        """).fetchall()

        distance_buckets = con.execute("""
            SELECT
                CASE
                    WHEN actual_km < 5  THEN 'under 5km'
                    WHEN actual_km < 10 THEN '5–10km'
                    WHEN actual_km < 15 THEN '10–15km'
                    WHEN actual_km < 21 THEN '15–21km'
                    ELSE '21km+'
                END AS bucket,
                COUNT(*) AS count
            FROM runs
            GROUP BY bucket
            ORDER BY MIN(actual_km)
        """).fetchall()

        by_requested_km = con.execute("""
            SELECT requested_km, COUNT(*) AS count
            FROM runs
            GROUP BY requested_km
            ORDER BY requested_km
        """).fetchall()

    return {
        "totals": dict(totals),
        "top_locations": [dict(r) for r in top_locations],
        "by_week": [dict(r) for r in by_week],
        "distance_distribution": [dict(r) for r in distance_buckets],
        "by_requested_km": [dict(r) for r in by_requested_km],
    }
