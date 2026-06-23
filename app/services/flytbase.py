import asyncio
import base64
import json
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

    Returns the full token response dict, shaped roughly as:
      {"access": {"token": str, ...}, "refresh": {"token": str, ...}}

    Each token object carries its own lifetime, but the exact field is not a
    documented contract — get_token_expiry derives the absolute expiry from
    whichever of `expires_in`, the JWT `exp` claim, or `expiry` is present.

    Raises httpx.HTTPStatusError on 401/403.
    """
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    async with httpx.AsyncClient() as client:
        response = await client.get(
            FLYTBASE_TOKEN_URL,
            headers={"Authorization": f"Basic {credentials}"},
            timeout=15.0,
        )
        response.raise_for_status()
        return response.json()


def _decode_jwt_exp(token: Optional[str]) -> Optional[int]:
    """
    Returns the `exp` (unix timestamp) claim from a JWT WITHOUT verifying the
    signature, or None if it cannot be read. Used only to derive token expiry.
    """
    if not token:
        return None
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload = parts[1] + "=" * (-len(parts[1]) % 4)  # restore base64 padding
        claims = json.loads(base64.urlsafe_b64decode(payload))
        exp = claims.get("exp")
        return int(exp) if exp is not None else None
    except (ValueError, TypeError, KeyError):
        return None


def get_token_expiry(token_data: dict) -> Optional[str]:
    """
    Returns a token's absolute expiry as an ISO 8601 UTC string, or None if it
    cannot be determined.

    The exact shape of the FlytBase token response is not contractually
    documented, so this tries every plausible source in turn and is correct
    regardless of which one the live API actually provides:
      1. `expires_in` (relative lifetime in seconds) -> now + expires_in.
      2. The JWT `exp` claim decoded from the token itself (access tokens are
         JWTs), which needs no separate expiry field at all.
      3. An absolute `expiry` field (ISO 8601 string or unix timestamp), the
         shape the original code assumed.

    The 5-min pull cadence plus the 2-min is_token_expired buffer means a wrong
    guess only costs an extra re-auth, never a failed pull.
    """
    if not token_data:
        return None
    expires_in = token_data.get("expires_in")
    if isinstance(expires_in, (int, float)):
        return (datetime.now(timezone.utc) + timedelta(seconds=expires_in)).isoformat()
    exp = _decode_jwt_exp(token_data.get("token"))
    if exp is not None:
        return datetime.fromtimestamp(exp, timezone.utc).isoformat()
    return _normalize_expiry(token_data.get("expiry"))


def _normalize_expiry(expiry) -> Optional[str]:
    """
    Coerces an absolute `expiry` value (ISO 8601 string or unix timestamp) into an
    ISO 8601 UTC string, or None if it is absent/unparseable.
    """
    if expiry is None:
        return None
    if isinstance(expiry, (int, float)):
        return datetime.fromtimestamp(expiry, timezone.utc).isoformat()
    try:
        parsed = datetime.fromisoformat(expiry)
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat()


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

def _unwrap(data):
    """
    Returns the telemetry payload, unwrapping the single-element array that some
    FlytBase topics wrap their payload in (`[ {...} ]` instead of `{...}`).

    Derived from FlytBase's reference telemetry example (`if (Array.isArray(d))
    d = d[0]`). That example is not committed to this repo — it is FlytBase's
    own integration sample; capture a copy under docs/ if one is needed.
    """
    if isinstance(data, list):
        return data[0] if data else {}
    return data


def average_location(positions: List[Tuple[dict, str]]) -> Optional[Tuple[float, float]]:
    """
    Returns the arithmetic-mean (lat, lon) across a list of (DroneGlobalPositionData,
    recorded_at) tuples, or None if none contain usable coordinates.

    Uses position.latitude / position.longitude; entries missing either are skipped.
    A simple mean is adequate for a drone operating within a local area — it is not
    antimeridian-aware (out of scope).
    """
    lats: List[float] = []
    lons: List[float] = []
    for payload, _recorded_at in positions:
        pos = (payload or {}).get("position") or {}
        lat = pos.get("latitude")
        lon = pos.get("longitude")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            lats.append(lat)
            lons.append(lon)
    if not lats:
        return None
    return (sum(lats) / len(lats), sum(lons) / len(lons))


def _drone_state_key(payload: dict) -> tuple:
    """Categorical state fields used to detect a drone_state transition."""
    mode = (payload or {}).get("mode") or {}
    return (
        (payload or {}).get("connected"),
        (payload or {}).get("armed"),
        mode.get("state"),
        (payload or {}).get("drone_state"),
    )


def reduce_drone_state(readings: List[Tuple[dict, str]]) -> List[Tuple[dict, str]]:
    """
    Collapses consecutive readings with an identical state key
    (connected, armed, mode.state, drone_state) into one entry per segment.

    Returns one (payload, recorded_at) per consecutive-unique state — payload is the
    first reading of the segment and recorded_at is its arrival time (the transition
    moment). Empty input -> []. A run that never changes -> a single entry.
    Transition criteria are defined by `_drone_state_key`; change that helper to alter what counts as a state change.
    """
    reduced: List[Tuple[dict, str]] = []
    prev_key = object()  # sentinel distinct from any real key
    for payload, recorded_at in readings:
        key = _drone_state_key(payload)
        if key != prev_key:
            reduced.append((payload, recorded_at))
            prev_key = key
    return reduced


def reduce_battery(readings: List[Tuple[dict, str]]) -> List[Tuple[dict, str]]:
    """
    Collapses consecutive readings with an identical charging_state into one entry per
    segment. Within each segment, total_percentage is replaced by the mean of the
    present numeric total_percentage values.

    Returns one (payload, recorded_at) per consecutive-unique charging_state, where
    recorded_at is the segment's first reading's arrival time. Empty input -> [].
    If a segment has no numeric total_percentage values, the first reading's total_percentage is kept unchanged (it may be None).
    """
    if not readings:
        return []

    segments: List[List[Tuple[dict, str]]] = []
    prev_key = object()
    for payload, recorded_at in readings:
        key = (payload or {}).get("charging_state")
        if not segments or key != prev_key:
            segments.append([])
            prev_key = key
        segments[-1].append((payload, recorded_at))

    reduced: List[Tuple[dict, str]] = []
    for seg in segments:
        first_payload, first_ts = seg[0]
        pcts: List[float] = []
        for payload, _recorded_at in seg:
            value = (payload or {}).get("total_percentage")
            if isinstance(value, (int, float)):
                pcts.append(value)
        merged = dict(first_payload or {})
        if pcts:
            merged["total_percentage"] = sum(pcts) / len(pcts)
        reduced.append((merged, first_ts))
    return reduced


def resolve_dock_for_drone(
    drone_id: str,
    dock_ids: Optional[List[str]],
    drone_dock_map: Optional[Dict[str, str]],
) -> Optional[str]:
    """
    Returns the dock_id to use for a drone's location fallback, or None.

    Resolution order:
      1. Explicit drone_dock_map[drone_id], if present.
      2. The sole configured dock, if exactly one dock_id is configured.
      3. None (no fallback available).
    """
    if drone_dock_map and drone_id in drone_dock_map:
        return drone_dock_map[drone_id]
    if dock_ids and len(dock_ids) == 1:
        return dock_ids[0]
    return None


async def collect_drone_telemetry(
    access_token: str,
    org_id: str,
    drone_ids: List[str],
    server_region: str,
    window_seconds: int,
    collect_battery: bool = True,
    collect_drone_state: bool = True,
    collect_notifications: bool = True,
) -> Dict[str, dict]:
    """
    Connects to the FlytBase Socket.IO endpoint, subscribes to each drone's
    global_position channel (always) plus battery / drone_state / notification
    (per flag), and collects all received messages for window_seconds.

    Returns:
        {
            drone_id: {
                "positions":    [(position_data, recorded_at_iso), ...],
                "battery":      [(battery_data, recorded_at_iso), ...],
                "drone_state":  [(state_data, recorded_at_iso), ...],
                "notification": [(notification_data, recorded_at_iso), ...],
            }
        }

    Each message is stamped with the client-side arrival time (UTC ISO 8601);
    FlytBase payloads carry no server-side timestamp.

    Raises on connection timeout. On connect_error the function returns empty
    per-drone buckets rather than raising (mirrors collect_dock_telemetry).

    NOTE: the Socket.IO lifecycle below — connect/connect_error/disconnect
    handlers, the per-topic "Subscribe" emit loop, and the connect → sleep →
    finally-disconnect block — is duplicated almost verbatim in
    collect_dock_telemetry. The two differ only in which channels/handlers they
    register. If this is touched again, consider extracting a shared
    _collect_over_socketio(topics, handler_registrar, ...) helper so the protocol
    lives in one place; kept separate for now to keep the change reviewable.
    """
    base_url = FLYTBASE_SOCKET_URLS[server_region.upper()]
    collected: Dict[str, dict] = {
        did: {"positions": [], "battery": [], "drone_state": [], "notification": []}
        for did in drone_ids
    }

    # Topic list mirrors the handlers registered below. global_position is always
    # subscribed (used for the window-average location); the rest are flag-gated.
    topics: List[str] = []
    for did in drone_ids:
        topics.append(f"{did}/global_position")
        if collect_battery:
            topics.append(f"{did}/battery")
        if collect_drone_state:
            topics.append(f"{did}/drone_state")
        if collect_notifications:
            topics.append(f"{did}/notification")

    connection_event = asyncio.Event()
    disconnect_event = asyncio.Event()

    sio = socketio.AsyncClient(logger=False, engineio_logger=False)

    @sio.event
    async def connect():
        # FlytBase telemetry channels do NOT auto-push: after connecting we must
        # emit a per-topic "Subscribe" (capital S) with payload
        # {"topic": "<id>/<channel>"}, one emit per topic. Per FlytBase's reference
        # telemetry example (their own sample, not committed to this repo).
        logger.info(f"FlytBase Socket.IO connected; subscribing to {len(topics)} topic(s).")
        for topic in topics:
            await sio.emit("Subscribe", {"topic": topic})
        connection_event.set()

    @sio.event
    async def connect_error(data):
        logger.error(f"FlytBase Socket.IO connection error: {data}")
        connection_event.set()  # unblock the wait so we can raise/return

    @sio.event
    async def disconnect():
        logger.info("FlytBase Socket.IO disconnected.")
        disconnect_event.set()

    # One handler factory for every channel — bucket selects the collected sub-list.
    def make_handler(did: str, bucket: str):
        async def handler(data):
            recorded_at = datetime.now(timezone.utc).isoformat()
            collected[did][bucket].append((_unwrap(data), recorded_at))
        return handler

    for drone_id in drone_ids:
        sio.on(f"{drone_id}/global_position", make_handler(drone_id, "positions"))
        if collect_battery:
            sio.on(f"{drone_id}/battery", make_handler(drone_id, "battery"))
        if collect_drone_state:
            sio.on(f"{drone_id}/drone_state", make_handler(drone_id, "drone_state"))
        if collect_notifications:
            sio.on(f"{drone_id}/notification", make_handler(drone_id, "notification"))

    try:
        await sio.connect(
            base_url,
            socketio_path=FLYTBASE_SOCKET_PATH,
            transports=["websocket"],
            auth={
                "authorization": f"Bearer {access_token}",
                "org-id": org_id,
            },
            wait_timeout=10,
        )
        await asyncio.wait_for(connection_event.wait(), timeout=15.0)
        logger.info(f"Collecting telemetry for {window_seconds}s from {len(drone_ids)} drone(s).")
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

    counts = ", ".join(
        f"{did}=(pos={len(v['positions'])}, batt={len(v['battery'])}, "
        f"state={len(v['drone_state'])}, notif={len(v['notification'])})"
        for did, v in collected.items()
    )
    logger.info(f"Drone collection complete. {counts}")
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

    # Build the topic list mirroring the handlers registered below. global_position
    # is always subscribed (needed for the dock's GPS coords); dock_state/weather
    # are gated by the collect_* flags.
    topics: List[str] = []
    for did in dock_ids:
        topics.append(f"{did}/global_position")
        if collect_dock_state:
            topics.append(f"{did}/dock_state")
        if collect_dock_weather:
            topics.append(f"{did}/weather")

    connection_event = asyncio.Event()
    disconnect_event = asyncio.Event()

    sio = socketio.AsyncClient(logger=False, engineio_logger=False)

    @sio.event
    async def connect():
        # FlytBase telemetry requires an explicit per-topic "Subscribe" emit after
        # connecting (capital S, payload {"topic": "<id>/<channel>"}, one per topic).
        # See collect_drone_telemetry; rationale per FlytBase's reference telemetry
        # example (their own sample, not committed to this repo).
        logger.info(f"FlytBase Socket.IO (dock) connected; subscribing to {len(topics)} topic(s).")
        for topic in topics:
            await sio.emit("Subscribe", {"topic": topic})
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
            async def handler(data):
                data = _unwrap(data)
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
                async def handler(data):
                    recorded_at = datetime.now(timezone.utc).isoformat()
                    collected[did]["dock_state"].append((_unwrap(data), recorded_at))
                return handler

            sio.on(f"{dock_id}/dock_state", make_state_handler(dock_id))

        if collect_dock_weather:
            def make_weather_handler(did: str):
                async def handler(data):
                    recorded_at = datetime.now(timezone.utc).isoformat()
                    collected[did]["weather"].append((_unwrap(data), recorded_at))
                return handler

            sio.on(f"{dock_id}/weather", make_weather_handler(dock_id))

    try:
        await sio.connect(
            base_url,
            socketio_path=FLYTBASE_SOCKET_PATH,
            transports=["websocket"],
            auth={
                "authorization": f"Bearer {access_token}",
                # Field is "org-id" per the AsyncAPI handshakeAuth spec. The docs
                # quick-start example shows "orgId", but the live server rejects
                # that with "organization undefined" (AUTH_FAILED).
                "org-id": org_id,
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


def transform_battery_to_observation(
    drone_id: str,
    battery_data: dict,
    recorded_at: str,
    lat: Optional[float],
    lon: Optional[float],
    subject_type: str = "drone",
    drone_name: Optional[str] = None,
) -> Optional[dict]:
    """
    Maps a (reduced) FlytBase battery payload to a Gundi observation dict.
    Returns None if lat or lon is missing.

    total_percentage is expected to already be the per-segment average (see
    reduce_battery). telemetry_type="battery" distinguishes it downstream.

    Field mapping:
        source           <- drone_id
        source_name      <- drone_name (falls back to drone_id)
        subject_type     <- subject_type config value
        recorded_at      <- segment start time (UTC ISO 8601)
        location.lat/lon <- window-average drone GPS (passed by caller)
        additional.*     <- telemetry_type, battery_percentage, charging_state
    """
    if lat is None or lon is None:
        logger.debug(f"Skipping battery for drone {drone_id}: missing location")
        return None
    return {
        "source": drone_id,
        "source_name": drone_name or drone_id,
        "subject_type": subject_type,
        "recorded_at": recorded_at,
        "location": {"lat": lat, "lon": lon},
        "additional": {
            "telemetry_type": "battery",
            "battery_percentage": battery_data.get("total_percentage"),
            "charging_state": battery_data.get("charging_state"),
        },
    }


def transform_drone_state_to_observation(
    drone_id: str,
    state_data: dict,
    recorded_at: str,
    lat: Optional[float],
    lon: Optional[float],
    subject_type: str = "drone",
    drone_name: Optional[str] = None,
) -> Optional[dict]:
    """
    Maps a (reduced) FlytBase drone_state payload to a Gundi observation dict.
    Returns None if lat or lon is missing. telemetry_type="drone_state".

    Field mapping:
        source           <- drone_id
        source_name      <- drone_name (falls back to drone_id)
        subject_type     <- subject_type config value
        recorded_at      <- transition time (UTC ISO 8601)
        location.lat/lon <- window-average drone GPS (passed by caller)
        additional.*     <- telemetry_type, connected, armed, mode_state, drone_state
    """
    if lat is None or lon is None:
        logger.debug(f"Skipping drone_state for drone {drone_id}: missing location")
        return None
    mode = (state_data.get("mode") or {})
    return {
        "source": drone_id,
        "source_name": drone_name or drone_id,
        "subject_type": subject_type,
        "recorded_at": recorded_at,
        "location": {"lat": lat, "lon": lon},
        "additional": {
            "telemetry_type": "drone_state",
            "connected": state_data.get("connected"),
            "armed": state_data.get("armed"),
            "mode_state": mode.get("state"),
            "drone_state": state_data.get("drone_state"),
        },
    }


def transform_notification_to_observation(
    drone_id: str,
    notification_data: dict,
    recorded_at: str,
    lat: Optional[float],
    lon: Optional[float],
    subject_type: str = "drone",
    drone_name: Optional[str] = None,
) -> Optional[dict]:
    """
    Maps a FlytBase notification payload to a Gundi observation dict.
    Returns None if lat or lon is missing. telemetry_type="notification".
    Optional message/code fields are included only when present.
    """
    if lat is None or lon is None:
        logger.debug(f"Skipping notification for drone {drone_id}: missing location")
        return None
    additional = {
        "telemetry_type": "notification",
        "level": notification_data.get("level"),
        "category": notification_data.get("category"),
    }
    message = notification_data.get("message")
    if message is not None:
        additional["message"] = message
    code = notification_data.get("code")
    if code is not None:
        additional["code"] = code
    return {
        "source": drone_id,
        "source_name": drone_name or drone_id,
        "subject_type": subject_type,
        "recorded_at": recorded_at,
        "location": {"lat": lat, "lon": lon},
        "additional": additional,
    }
