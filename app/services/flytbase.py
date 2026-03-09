import asyncio
import base64
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import httpx
import socketio

logger = logging.getLogger(__name__)

FLYTBASE_TOKEN_URL = "https://api.flytbase.com/oauth/token"
FLYTBASE_SOCKET_URLS = {
    "US": "wss://api.flytbase.com",
    "EU": "wss://api-eu.flytbase.com",
}
FLYTBASE_SOCKET_PATH = "/socket/socket.io"


# ── OAuth ────────────────────────────────────────────────────────────────────

async def get_flytbase_token(client_id: str, client_secret: str) -> dict:
    """
    Performs OAuth2 Client Credentials flow against the FlytBase token endpoint.

    Returns the full token response dict:
      {"access": {"token": str, "expiry": str}, "refresh": {"token": str, "expiry": str}}

    Raises httpx.HTTPStatusError on 401/403.
    """
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    async with httpx.AsyncClient() as client:
        response = await client.post(
            FLYTBASE_TOKEN_URL,
            headers={"Authorization": f"Basic {credentials}"},
            timeout=15.0,
        )
        response.raise_for_status()
        return response.json()


async def refresh_flytbase_token(refresh_token: str) -> dict:
    """
    Uses a refresh token to obtain a new access token.

    NOTE: The exact FlytBase refresh grant type is assumed to be standard OAuth2
    (grant_type=refresh_token). Verify against FlytBase docs if this fails.

    Returns the same dict shape as get_flytbase_token.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            FLYTBASE_TOKEN_URL,
            data={"grant_type": "refresh_token", "refresh_token": refresh_token},
            timeout=15.0,
        )
        response.raise_for_status()
        return response.json()


def is_token_expired(expiry_iso: Optional[str], buffer_seconds: int = 120) -> bool:
    """
    Returns True if the token will expire within buffer_seconds from now.

    A 2-minute buffer (default) guards against token expiry mid-collection.
    Returns True for None or unparseable values as a fail-safe.
    """
    if not expiry_iso:
        return True
    try:
        expiry = datetime.fromisoformat(expiry_iso)
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) >= (expiry - timedelta(seconds=buffer_seconds))
    except (ValueError, TypeError):
        return True


# ── Socket.IO collection ─────────────────────────────────────────────────────

async def collect_drone_positions(
    access_token: str,
    org_id: str,
    drone_ids: List[str],
    server_region: str,
    window_seconds: int,
) -> Dict[str, List[Tuple[dict, str]]]:
    """
    Connects to the FlytBase Socket.IO endpoint, subscribes to each drone's
    global_position channel, and collects all received messages for window_seconds.

    Returns:
        {drone_id: [(position_data, recorded_at_iso), ...]}

    Each message is stamped with the client-side arrival time (UTC ISO 8601).
    The FlytBase global_position payload does not include a server-side timestamp,
    so arrival time is the best available approximation for recorded_at.

    Raises on connection failure (socketio/aiohttp exception propagates up).
    """
    base_url = FLYTBASE_SOCKET_URLS[server_region.upper()]
    collected: Dict[str, List[Tuple[dict, str]]] = {did: [] for did in drone_ids}

    connection_event = asyncio.Event()
    disconnect_event = asyncio.Event()

    sio = socketio.AsyncClient(logger=False, engineio_logger=False)

    @sio.event
    async def connect():
        logger.info("FlytBase Socket.IO connected.")
        connection_event.set()

    @sio.event
    async def connect_error(data):
        logger.error(f"FlytBase Socket.IO connection error: {data}")
        connection_event.set()  # unblock the wait so we can raise/return

    @sio.event
    async def disconnect():
        logger.info("FlytBase Socket.IO disconnected.")
        disconnect_event.set()

    # Register a handler per drone — use closure to capture drone_id correctly
    for drone_id in drone_ids:
        channel = f"{drone_id}/global_position"

        def make_handler(did: str):
            async def handler(data: dict):
                recorded_at = datetime.now(timezone.utc).isoformat()
                collected[did].append((data, recorded_at))
            return handler

        sio.on(channel, make_handler(drone_id))

    try:
        await sio.connect(
            base_url,
            socketio_path=FLYTBASE_SOCKET_PATH,
            transports=["websocket"],
            auth={
                "authorization": f"Bearer {access_token}",
                "orgId": org_id,
            },
            wait_timeout=10,
        )
        await asyncio.wait_for(connection_event.wait(), timeout=15.0)
        logger.info(f"Collecting positions for {window_seconds}s from {len(drone_ids)} drone(s).")
        await asyncio.sleep(window_seconds)
    except asyncio.TimeoutError:
        logger.error("Timed out waiting for FlytBase Socket.IO connection.")
        raise
    finally:
        if sio.connected:
            await sio.disconnect()
        try:
            await asyncio.wait_for(disconnect_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass  # best-effort graceful disconnect

    counts = ", ".join(f"{did}={len(msgs)}" for did, msgs in collected.items())
    logger.info(f"Collection complete. Messages per drone: {counts}")
    return collected


# ── Socket.IO dock collection ─────────────────────────────────────────────────

async def collect_dock_telemetry(
    access_token: str,
    org_id: str,
    dock_ids: List[str],
    server_region: str,
    window_seconds: int,
    collect_dock_state: bool = True,
    collect_dock_weather: bool = True,
) -> Dict[str, dict]:
    """
    Connects to the FlytBase Socket.IO endpoint, subscribes to each dock's
    dock_state, weather, and global_position channels, and collects all received
    messages for window_seconds.

    Returns:
        {
            dock_id: {
                "dock_state": [(state_data, recorded_at_iso), ...],
                "weather":    [(weather_data, recorded_at_iso), ...],
                "dock_location": (lat, lon) | None,  # first global_position received
            }
        }

    global_position is always subscribed to obtain the dock's GPS coordinates,
    which are needed as the location field on dock_state and weather observations.
    If no global_position message arrives during the window, dock_location is None
    and the caller should skip observations for that dock.

    collect_dock_state and collect_dock_weather control whether those channels
    are subscribed; global_position is always subscribed regardless.
    """
    base_url = FLYTBASE_SOCKET_URLS[server_region.upper()]
    collected: Dict[str, dict] = {
        did: {"dock_state": [], "weather": [], "dock_location": None}
        for did in dock_ids
    }

    connection_event = asyncio.Event()
    disconnect_event = asyncio.Event()

    sio = socketio.AsyncClient(logger=False, engineio_logger=False)

    @sio.event
    async def connect():
        logger.info("FlytBase Socket.IO (dock) connected.")
        connection_event.set()

    @sio.event
    async def connect_error(data):
        logger.error(f"FlytBase Socket.IO (dock) connection error: {data}")
        connection_event.set()

    @sio.event
    async def disconnect():
        logger.info("FlytBase Socket.IO (dock) disconnected.")
        disconnect_event.set()

    for dock_id in dock_ids:
        # Always subscribe to global_position for dock lat/lon
        def make_position_handler(did: str):
            async def handler(data: dict):
                if collected[did]["dock_location"] is None:
                    loc = (data.get("dock_location") or {})
                    lat = loc.get("latitude")
                    lon = loc.get("longitude")
                    if lat is not None and lon is not None:
                        collected[did]["dock_location"] = (lat, lon)
            return handler

        sio.on(f"{dock_id}/global_position", make_position_handler(dock_id))

        if collect_dock_state:
            def make_state_handler(did: str):
                async def handler(data: dict):
                    recorded_at = datetime.now(timezone.utc).isoformat()
                    collected[did]["dock_state"].append((data, recorded_at))
                return handler

            sio.on(f"{dock_id}/dock_state", make_state_handler(dock_id))

        if collect_dock_weather:
            def make_weather_handler(did: str):
                async def handler(data: dict):
                    recorded_at = datetime.now(timezone.utc).isoformat()
                    collected[did]["weather"].append((data, recorded_at))
                return handler

            sio.on(f"{dock_id}/weather", make_weather_handler(dock_id))

    try:
        await sio.connect(
            base_url,
            socketio_path=FLYTBASE_SOCKET_PATH,
            transports=["websocket"],
            auth={
                "authorization": f"Bearer {access_token}",
                "orgId": org_id,
            },
            wait_timeout=10,
        )
        await asyncio.wait_for(connection_event.wait(), timeout=15.0)
        logger.info(f"Collecting dock telemetry for {window_seconds}s from {len(dock_ids)} dock(s).")
        await asyncio.sleep(window_seconds)
    except asyncio.TimeoutError:
        logger.error("Timed out waiting for FlytBase Socket.IO (dock) connection.")
        raise
    finally:
        if sio.connected:
            await sio.disconnect()
        try:
            await asyncio.wait_for(disconnect_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass

    counts = ", ".join(
        f"{did}=(state={len(v['dock_state'])}, weather={len(v['weather'])}, loc={'ok' if v['dock_location'] else 'missing'})"
        for did, v in collected.items()
    )
    logger.info(f"Dock collection complete. {counts}")
    return collected


# ── Observation transformation ────────────────────────────────────────────────

def transform_position_to_observation(
    drone_id: str,
    position_data: dict,
    recorded_at: str,
    subject_type: str = "drone",
    drone_name: Optional[str] = None,
) -> Optional[dict]:
    """
    Maps a FlytBase DroneGlobalPositionData dict to a Gundi observation dict.

    Returns None if latitude or longitude is missing (caller should filter these out).

    Field mapping:
        source           ← drone_id
        source_name      ← drone_name (falls back to drone_id)
        subject_type     ← subject_type config value
        recorded_at      ← client-side message arrival time (UTC ISO 8601)
        location.lat     ← position.latitude
        location.lon     ← position.longitude
        additional.*     ← elevation, height, speed, RTK quality, home distance
    """
    position = position_data.get("position") or {}
    lat = position.get("latitude")
    lon = position.get("longitude")

    if lat is None or lon is None:
        logger.debug(f"Skipping position for drone {drone_id}: missing lat/lon")
        return None

    speed = position_data.get("speed") or {}
    home = position_data.get("home_position") or {}
    rtk = position_data.get("rtk") or {}

    horizontal_ms = speed.get("horizontal")
    speed_kmph = round(horizontal_ms * 3.6, 2) if horizontal_ms is not None else None

    return {
        "source": drone_id,
        "source_name": drone_name or drone_id,
        "subject_type": subject_type,
        "recorded_at": recorded_at,
        "location": {
            "lat": lat,
            "lon": lon,
        },
        "additional": {
            "elevation_msl_m": position.get("elevation"),
            "height_above_takeoff_m": position.get("height"),
            "gps_satellites": position.get("gps_satellites"),
            "speed_horizontal_ms": horizontal_ms,
            "speed_vertical_ms": speed.get("vertical"),
            "speed_kmph": speed_kmph,
            "home_distance_m": home.get("distance"),
            "rtk_quality": rtk.get("quality"),
            "rtk_satellites": rtk.get("rtk_satellites"),
            "rtk_fix_state": rtk.get("fix_state"),
        },
    }


def transform_dock_state_to_observation(
    dock_id: str,
    state_data: dict,
    recorded_at: str,
    dock_lat: Optional[float],
    dock_lon: Optional[float],
    subject_type: str = "dock",
    dock_name: Optional[str] = None,
) -> Optional[dict]:
    """
    Maps a FlytBase DockStateEventData dict to a Gundi observation dict.

    Returns None if dock_lat or dock_lon is missing (no GPS fix yet for this dock).

    Field mapping:
        source           ← dock_id
        source_name      ← dock_name (falls back to dock_id)
        subject_type     ← subject_type config value
        recorded_at      ← client-side message arrival time (UTC ISO 8601)
        location.lat/lon ← dock GPS coords from {dockId}/global_position
        additional.*     ← dock_state, enclosure, charging_rods, drone_charging,
                           drone_power, emergency_stop, occupancy, connected,
                           operation_mode, total_flight_operations
    """
    if dock_lat is None or dock_lon is None:
        logger.debug(f"Skipping dock_state for dock {dock_id}: missing dock location")
        return None

    enclosure = (state_data.get("enclosure") or {})
    charging_rods = (state_data.get("charging_rods") or {})
    drone_charging = (state_data.get("drone_charging") or {})
    drone_power = (state_data.get("drone_power") or {})
    emergency_stop = (state_data.get("emergency_stop") or {})
    occupancy = (state_data.get("occupancy") or {})
    operation_mode = (state_data.get("operation_mode") or {})

    return {
        "source": dock_id,
        "source_name": dock_name or dock_id,
        "subject_type": subject_type,
        "recorded_at": recorded_at,
        "location": {
            "lat": dock_lat,
            "lon": dock_lon,
        },
        "additional": {
            "dock_state": state_data.get("dock_state"),
            "enclosure_state": enclosure.get("state"),
            "charging_rods_state": charging_rods.get("state"),
            "drone_charging_state": drone_charging.get("state"),
            "drone_power_state": drone_power.get("state"),
            "emergency_stop_state": emergency_stop.get("state"),
            "occupancy_state": occupancy.get("state"),
            "connected": state_data.get("connected"),
            "operation_mode_state": operation_mode.get("state"),
            "total_flight_operations": state_data.get("total_flight_operations"),
        },
    }


def transform_dock_weather_to_observation(
    dock_id: str,
    weather_data: dict,
    recorded_at: str,
    dock_lat: Optional[float],
    dock_lon: Optional[float],
    subject_type: str = "dock",
    dock_name: Optional[str] = None,
) -> Optional[dict]:
    """
    Maps a FlytBase DockWeatherEventData dict to a Gundi observation dict.

    Returns None if dock_lat or dock_lon is missing.

    Field mapping:
        source           ← dock_id
        source_name      ← dock_name (falls back to dock_id)
        subject_type     ← subject_type config value
        recorded_at      ← client-side message arrival time (UTC ISO 8601)
        location.lat/lon ← dock GPS coords from {dockId}/global_position
        additional.*     ← humidity_pct, rainfall, temperature_c,
                           wind_speed_ms, wind_direction_deg
    """
    if dock_lat is None or dock_lon is None:
        logger.debug(f"Skipping weather for dock {dock_id}: missing dock location")
        return None

    weather = (weather_data.get("weather") or {})
    wind = (weather.get("wind") or {})

    return {
        "source": dock_id,
        "source_name": dock_name or dock_id,
        "subject_type": subject_type,
        "recorded_at": recorded_at,
        "location": {
            "lat": dock_lat,
            "lon": dock_lon,
        },
        "additional": {
            "humidity_pct": weather.get("humidity"),
            "rainfall": weather.get("rainfall"),
            "temperature_c": weather.get("temperature"),
            "wind_speed_ms": wind.get("speed"),
            "wind_direction_deg": wind.get("direction"),
        },
    }
