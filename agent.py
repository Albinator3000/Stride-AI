import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()

import groq
from groq import Groq
from tools import geocode_location, plan_running_route, save_run_plan, export_route_gpx

logger = logging.getLogger(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You are a running route assistant. Treat the requested distance as a hard training requirement — the user needs to hit that distance exactly or slightly over (up to 0.3km). A short route is not acceptable.

When the user gives a start location and a desired distance:

First, call the geocode tool with the address exactly as given. Do not alter or abbreviate it.

Second, call the route planning tool. For lat and lon, copy the exact decimal values returned by the geocode tool — do not round or truncate them. Use the user's requested distance and the original address string. If the user mentions terrain or scenery preferences (e.g. "riverside", "through the park", "scenic", "quiet", "forest trails", "hilly"), extract the matching preference keywords and pass them as the preferences array. Valid values: park, river, scenic, trail, green, forest, quiet, hilly, riverside, lake, water.

Third, call the save tool. Take every value directly from the route planning result — total_distance_km, elevation_gain_m, steps (the full list), map_path, and route_id. Do not substitute, invent, or paraphrase any of these. Use the user's requested distance for distance_km, and write a short title like "Battersea Power Station 10km".

Finally, reply with this exact structure:

Route: {total_distance_km}km | Elevation: {elevation_gain_m}m gain
Map: {map_path}

Directions:
{steps from the route result, one per line, copied verbatim — every step, in order, nothing omitted or rewritten}

---
{one sentence offering an alternate route via a different seed}

Important rules:
- Never round or truncate lat/lon values between tool calls
- Never invent any value — all numbers and directions must come from tool results
- Do not estimate finish time
- If the route result contains a distance_warning field, show it before the directions
- If any tool returns an error status, stop immediately and report the exact error message
- If the user asks to export a route to GPX (for Garmin, Wahoo, etc.), call export_route_gpx with the route_id from the most recent plan_running_route result and report the gpx_path"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "geocode_location",
            "description": "Convert a human-readable address or landmark into latitude and longitude coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {
                        "type": "string",
                        "description": "The address, landmark, or place name to geocode.",
                    }
                },
                "required": ["address"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_running_route",
            "description": "Generate a round-trip running route and create an interactive HTML map. Returns directions, actual distance, elevation gain, map file path, and OSM URL. Do NOT pass coordinates back to the user — just use the returned values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "description": "Start latitude."},
                    "lon": {"type": "number", "description": "Start longitude."},
                    "distance_km": {
                        "type": "number",
                        "description": "Target total running distance in kilometers.",
                    },
                    "start_address": {
                        "type": "string",
                        "description": "Human-readable start location name (used in the map marker).",
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Route variation seed. Use 1 for the default route, 2+ for alternates.",
                        "default": 1,
                    },
                    "preferences": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional terrain preferences extracted from the user's message. "
                            "Valid values: park, river, scenic, trail, green, forest, quiet, "
                            "hilly, riverside, lake, water. Omit if the user has no preference."
                        ),
                    },
                },
                "required": ["lat", "lon", "distance_km", "start_address"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "export_route_gpx",
            "description": "Export a saved route as a GPX file for use with Garmin, Wahoo, or any GPS device/app.",
            "parameters": {
                "type": "object",
                "properties": {
                    "route_id": {"type": "string", "description": "route_id from plan_running_route."},
                    "title": {"type": "string", "description": "Name for the GPX track (e.g. 'Battersea 10km')."},
                },
                "required": ["route_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_run_plan",
            "description": "Save the run plan and route details to a JSON file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "A short title for this run."},
                    "location": {"type": "string", "description": "Start location name."},
                    "distance_km": {"type": "number", "description": "Requested distance in km."},
                    "total_distance_km": {"type": "number", "description": "Actual route distance in km."},
                    "elevation_gain_m": {"type": "number", "description": "Total elevation gain in meters."},
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Turn-by-turn direction strings from plan_running_route.",
                    },
                    "map_path": {"type": "string", "description": "Map file path from plan_running_route."},
                    "route_id": {"type": "string", "description": "route_id returned by plan_running_route, links to stored coordinates."},
                },
                "required": [
                    "title", "location", "distance_km", "total_distance_km",
                    "elevation_gain_m", "steps", "map_path", "route_id",
                ],
            },
        },
    },
]

TOOL_FN_MAP = {
    "geocode_location": geocode_location,
    "plan_running_route": plan_running_route,
    "save_run_plan": save_run_plan,
    "export_route_gpx": export_route_gpx,
}


def _build_messages(user_message: str, conversation_history: list | None) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})
    return messages


def _dispatch_tool(call) -> dict:
    fn_name = call.function.name
    fn_args = json.loads(call.function.arguments)
    logger.debug("tool call: %s(%s)", fn_name, fn_args)
    result = TOOL_FN_MAP[fn_name](**fn_args)
    logger.debug("tool result: %s", {k: v for k, v in result.items() if k != "steps"})
    return fn_name, fn_args, result


def run_agent(user_message: str, conversation_history: list | None = None) -> str:
    messages = _build_messages(user_message, conversation_history)

    try:
        while True:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            choice = response.choices[0]
            msg = choice.message

            if choice.finish_reason == "stop":
                return msg.content

            messages.append(msg)

            for call in msg.tool_calls:
                _, _, result = _dispatch_tool(call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result),
                    }
                )
    except groq.RateLimitError:
        raise RuntimeError("Groq rate limit reached — please wait a moment and try again.")
    except groq.BadRequestError as e:
        raise RuntimeError(f"Bad request to Groq API: {e}")


def stream_agent(user_message: str, conversation_history: list | None = None):
    """
    Generator that yields SSE-formatted strings.
    Events: tool_call, tool_result, token, done, error.
    """
    messages = _build_messages(user_message, conversation_history)

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    try:
        while True:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            choice = response.choices[0]
            msg = choice.message

            if choice.finish_reason == "stop":
                content = msg.content or ""
                for word in content.split(" "):
                    yield sse("token", {"text": word + " "})
                yield sse("done", {"full_text": content})
                return

            messages.append(msg)

            for call in msg.tool_calls:
                fn_name, fn_args, result = _dispatch_tool(call)
                yield sse("tool_call", {"tool": fn_name, "args": fn_args})
                yield sse("tool_result", {"tool": fn_name, "status": result.get("status", "ok")})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result),
                    }
                )
    except groq.RateLimitError:
        yield sse("error", {"message": "Groq rate limit reached — please wait a moment and try again."})
    except groq.BadRequestError as e:
        yield sse("error", {"message": f"Bad request to Groq API: {e}"})


def main():
    print("Stride AI — Running Route Planner")
    print("Type your run request (e.g. '5km from Central Park') or 'quit' to exit.\n")
    history = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        reply = run_agent(user_input, history)
        print(f"\nAgent: {reply}\n")
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
