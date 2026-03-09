import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services import flytbase


# ── Fixtures ──────────────────────────────────────────────────────────────────

DRONE_ID = "648f2a3d7b1c9e5f4a8d0c2e"
RECORDED_AT = "2024-01-24T09:03:00+00:00"


@pytest.fixture
def sample_position_full():
    return {
        "position": {
            "latitude": 18.5664295,
            "longitude": 73.7719138,
            "elevation": 550.0,
            "height": 25.3,
            "gps_satellites": 12,
        },
        "speed": {"horizontal": 4.17, "vertical": 0.5},
        "home_position": {"latitude": 18.566, "longitude": 73.7715, "distance": 56.2},
        "rtk": {"quality": 2, "rtk_satellites": 8, "fix_state": 2},
    }


@pytest.fixture
def sample_position_minimal():
    """Payload where all optional fields are None."""
    return {
        "position": {
            "latitude": 18.5664295,
            "longitude": 73.7719138,
            "elevation": None,
            "height": 0,
            "gps_satellites": None,
        },
        "speed": {"horizontal": None, "vertical": None},
        "home_position": {"latitude": None, "longitude": None, "distance": None},
        "rtk": {"quality": None, "rtk_satellites": None, "fix_state": None},
    }


@pytest.fixture
def sample_position_missing_coords():
    """Payload where lat/lon are absent — should be filtered."""
    return {
        "position": {"latitude": None, "longitude": None, "height": 0},
        "speed": {},
        "home_position": {},
        "rtk": {},
    }


@pytest.fixture
def mock_token_response():
    future = (datetime.now(timezone.utc) + timedelta(minutes=14)).isoformat()
    refresh_future = (datetime.now(timezone.utc) + timedelta(days=6)).isoformat()
    return {
        "access": {"token": "test-access-token", "expiry": future},
        "refresh": {"token": "test-refresh-token", "expiry": refresh_future},
    }


# ── OAuth: get_flytbase_token ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_flytbase_token_success(mocker, mock_token_response):
    mock_response = MagicMock()
    mock_response.json.return_value = mock_token_response
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
    mocker.patch("app.services.flytbase.httpx.AsyncClient", return_value=mock_client)

    result = await flytbase.get_flytbase_token("my-client-id", "my-secret")

    assert result["access"]["token"] == "test-access-token"
    assert result["refresh"]["token"] == "test-refresh-token"

    # Verify Basic auth header was constructed correctly
    import base64
    expected_b64 = base64.b64encode(b"my-client-id:my-secret").decode()
    call_kwargs = mock_client.__aenter__.return_value.post.call_args
    assert f"Basic {expected_b64}" in str(call_kwargs)


@pytest.mark.asyncio
async def test_get_flytbase_token_raises_on_http_error(mocker):
    import httpx

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "401 Unauthorized",
        request=MagicMock(),
        response=MagicMock(status_code=401),
    )

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
    mocker.patch("app.services.flytbase.httpx.AsyncClient", return_value=mock_client)

    with pytest.raises(httpx.HTTPStatusError):
        await flytbase.get_flytbase_token("bad-id", "bad-secret")


# ── Token expiry checks ───────────────────────────────────────────────────────

def test_is_token_expired_fresh_token():
    future = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    assert flytbase.is_token_expired(future) is False


def test_is_token_expired_within_buffer():
    # 90 seconds from now — within the default 120s buffer
    near = (datetime.now(timezone.utc) + timedelta(seconds=90)).isoformat()
    assert flytbase.is_token_expired(near) is True


def test_is_token_expired_exactly_at_buffer():
    # Exactly 120 seconds from now — should be treated as expired
    at_buffer = (datetime.now(timezone.utc) + timedelta(seconds=120)).isoformat()
    assert flytbase.is_token_expired(at_buffer) is True


def test_is_token_expired_none():
    assert flytbase.is_token_expired(None) is True


def test_is_token_expired_malformed():
    assert flytbase.is_token_expired("not-a-date") is True


def test_is_token_expired_custom_buffer():
    # 30 seconds from now, buffer=0 → not expired
    soon = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
    assert flytbase.is_token_expired(soon, buffer_seconds=0) is False


# ── Observation transformation ────────────────────────────────────────────────

def test_transform_full_position(sample_position_full):
    obs = flytbase.transform_position_to_observation(
        drone_id=DRONE_ID,
        position_data=sample_position_full,
        recorded_at=RECORDED_AT,
        subject_type="drone",
        drone_name="Alpha 1",
    )

    assert obs is not None
    assert obs["source"] == DRONE_ID
    assert obs["source_name"] == "Alpha 1"
    assert obs["subject_type"] == "drone"
    assert obs["recorded_at"] == RECORDED_AT
    assert obs["location"] == {"lat": 18.5664295, "lon": 73.7719138}
    assert obs["additional"]["elevation_msl_m"] == 550.0
    assert obs["additional"]["height_above_takeoff_m"] == 25.3
    assert obs["additional"]["gps_satellites"] == 12
    assert obs["additional"]["speed_horizontal_ms"] == 4.17
    assert obs["additional"]["speed_vertical_ms"] == 0.5
    assert obs["additional"]["speed_kmph"] == round(4.17 * 3.6, 2)
    assert obs["additional"]["home_distance_m"] == 56.2
    assert obs["additional"]["rtk_quality"] == 2
    assert obs["additional"]["rtk_satellites"] == 8
    assert obs["additional"]["rtk_fix_state"] == 2


def test_transform_minimal_position_no_crash(sample_position_minimal):
    obs = flytbase.transform_position_to_observation(
        drone_id=DRONE_ID,
        position_data=sample_position_minimal,
        recorded_at=RECORDED_AT,
    )

    assert obs is not None
    assert obs["location"]["lat"] == 18.5664295
    assert obs["location"]["lon"] == 73.7719138
    assert obs["additional"]["elevation_msl_m"] is None
    assert obs["additional"]["speed_horizontal_ms"] is None
    assert obs["additional"]["speed_kmph"] is None
    assert obs["additional"]["rtk_quality"] is None


def test_transform_missing_coords_returns_none(sample_position_missing_coords):
    obs = flytbase.transform_position_to_observation(
        drone_id=DRONE_ID,
        position_data=sample_position_missing_coords,
        recorded_at=RECORDED_AT,
    )
    assert obs is None


def test_transform_default_source_name(sample_position_full):
    obs = flytbase.transform_position_to_observation(
        drone_id=DRONE_ID,
        position_data=sample_position_full,
        recorded_at=RECORDED_AT,
    )
    assert obs["source_name"] == DRONE_ID


def test_transform_empty_nested_dicts():
    """Position data with completely missing sub-dicts should not raise."""
    obs = flytbase.transform_position_to_observation(
        drone_id=DRONE_ID,
        position_data={"position": {"latitude": 10.0, "longitude": 20.0}},
        recorded_at=RECORDED_AT,
    )
    assert obs is not None
    assert obs["additional"]["speed_horizontal_ms"] is None
    assert obs["additional"]["home_distance_m"] is None
    assert obs["additional"]["rtk_quality"] is None


# ── Socket.IO collection ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_collect_connects_with_correct_auth(mocker):
    mock_sio = AsyncMock()
    mock_sio.connected = True
    mock_sio.connect = AsyncMock()
    mock_sio.disconnect = AsyncMock()

    # Capture registered handlers so we can fire the connect event
    registered_events = {}

    def fake_event(func):
        registered_events[func.__name__] = func
        return func

    def fake_on(channel, handler):
        registered_events[channel] = handler

    mock_sio.event = fake_event
    mock_sio.on = fake_on

    mocker.patch("app.services.flytbase.socketio.AsyncClient", return_value=mock_sio)

    # Simulate the connect event firing after sio.connect() is called
    original_connect = mock_sio.connect

    async def connect_and_fire(*args, **kwargs):
        await original_connect(*args, **kwargs)
        # Fire the connect event handler to unblock connection_event
        if "connect" in registered_events:
            await registered_events["connect"]()

    mock_sio.connect = connect_and_fire
    mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    await flytbase.collect_drone_positions(
        access_token="test-token",
        org_id="test-org-id",
        drone_ids=[DRONE_ID],
        server_region="US",
        window_seconds=1,
    )

    original_connect.assert_called_once()
    call_kwargs = original_connect.call_args.kwargs
    assert call_kwargs["auth"]["authorization"] == "Bearer test-token"
    assert call_kwargs["auth"]["orgId"] == "test-org-id"
    assert call_kwargs["transports"] == ["websocket"]
    assert "api.flytbase.com" in original_connect.call_args.args[0]


@pytest.mark.asyncio
async def test_collect_registers_handler_per_drone(mocker):
    mock_sio = AsyncMock()
    mock_sio.connected = True
    mock_sio.connect = AsyncMock()
    mock_sio.disconnect = AsyncMock()

    registered_events = {}

    def fake_event(func):
        registered_events[func.__name__] = func
        return func

    registered_channels = []

    def fake_on(channel, handler):
        registered_channels.append(channel)

    mock_sio.event = fake_event
    mock_sio.on = fake_on

    mocker.patch("app.services.flytbase.socketio.AsyncClient", return_value=mock_sio)

    original_connect = mock_sio.connect

    async def connect_and_fire(*args, **kwargs):
        await original_connect(*args, **kwargs)
        if "connect" in registered_events:
            await registered_events["connect"]()

    mock_sio.connect = connect_and_fire
    mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    drone_ids = ["drone-001", "drone-002", "drone-003"]
    await flytbase.collect_drone_positions(
        access_token="tok",
        org_id="org",
        drone_ids=drone_ids,
        server_region="US",
        window_seconds=1,
    )

    for did in drone_ids:
        assert f"{did}/global_position" in registered_channels


@pytest.mark.asyncio
async def test_collect_returns_empty_when_no_messages(mocker):
    mock_sio = AsyncMock()
    mock_sio.connected = True
    mock_sio.connect = AsyncMock()
    mock_sio.disconnect = AsyncMock()

    registered_events = {}

    def fake_event(func):
        registered_events[func.__name__] = func
        return func

    mock_sio.event = fake_event
    mock_sio.on = MagicMock()

    mocker.patch("app.services.flytbase.socketio.AsyncClient", return_value=mock_sio)

    original_connect = mock_sio.connect

    async def connect_and_fire(*args, **kwargs):
        await original_connect(*args, **kwargs)
        if "connect" in registered_events:
            await registered_events["connect"]()

    mock_sio.connect = connect_and_fire
    mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    result = await flytbase.collect_drone_positions(
        access_token="tok",
        org_id="org",
        drone_ids=[DRONE_ID],
        server_region="EU",
        window_seconds=1,
    )

    assert isinstance(result, dict)
    assert DRONE_ID in result
    assert result[DRONE_ID] == []
    # EU server URL used
    original_connect.assert_called_once()
    assert "api-eu.flytbase.com" in original_connect.call_args.args[0]


# ── Dock telemetry fixtures ───────────────────────────────────────────────────

DOCK_ID = "507f1f77bcf86cd799439022"


@pytest.fixture
def sample_dock_global_position():
    return {
        "dock_location": {"latitude": 18.5628, "longitude": 73.7010, "altitude": 505.9},
        "safe_location": {"latitude": 18.5628, "longitude": 73.7010, "altitude": 0, "land_height": 40},
        "positioning": {"gps_satellites": 7, "rtk_satellites": 38, "fix_state": 2, "calibration_state": 1, "quality": 5},
    }


@pytest.fixture
def sample_dock_state():
    return {
        "dock_state": 1,
        "enclosure": {"state": 0},
        "charging_rods": {"state": 1},
        "drone_charging": {"state": 1},
        "drone_power": {"state": 1},
        "emergency_stop": {"state": 0},
        "occupancy": {"state": 1},
        "operation_mode": {"state": 1},
        "connected": True,
        "total_flight_operations": 42,
    }


@pytest.fixture
def sample_dock_weather():
    return {
        "weather": {
            "humidity": 65.0,
            "rainfall": 0,
            "temperature": 28.5,
            "wind": {"direction": 180, "speed": 3.2},
        }
    }


# ── collect_dock_telemetry helpers ────────────────────────────────────────────

def _make_dock_sio_mock(mocker, registered_events, registered_channels):
    """Creates a mock Socket.IO client and patches socketio.AsyncClient."""
    mock_sio = AsyncMock()
    mock_sio.connected = True
    mock_sio.connect = AsyncMock()
    mock_sio.disconnect = AsyncMock()

    def fake_event(func):
        registered_events[func.__name__] = func
        return func

    def fake_on(channel, handler):
        registered_channels.append(channel)

    mock_sio.event = fake_event
    mock_sio.on = fake_on

    mocker.patch("app.services.flytbase.socketio.AsyncClient", return_value=mock_sio)

    original_connect = mock_sio.connect

    async def connect_and_fire(*args, **kwargs):
        await original_connect(*args, **kwargs)
        if "connect" in registered_events:
            await registered_events["connect"]()

    mock_sio.connect = connect_and_fire
    return mock_sio, original_connect


# ── collect_dock_telemetry tests ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_collect_dock_telemetry_connects_with_correct_auth(mocker):
    registered_events = {}
    registered_channels = []
    _, original_connect = _make_dock_sio_mock(mocker, registered_events, registered_channels)
    mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    await flytbase.collect_dock_telemetry(
        access_token="dock-token",
        org_id="org-123",
        dock_ids=[DOCK_ID],
        server_region="US",
        window_seconds=1,
    )

    original_connect.assert_called_once()
    call_kwargs = original_connect.call_args.kwargs
    assert call_kwargs["auth"]["authorization"] == "Bearer dock-token"
    assert call_kwargs["auth"]["orgId"] == "org-123"
    assert call_kwargs["transports"] == ["websocket"]
    assert "api.flytbase.com" in original_connect.call_args.args[0]


@pytest.mark.asyncio
async def test_collect_dock_telemetry_registers_channels_per_dock(mocker):
    registered_events = {}
    registered_channels = []
    _make_dock_sio_mock(mocker, registered_events, registered_channels)
    mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    dock_ids = [DOCK_ID, "62a8c3b4d5e9f12345678902"]
    await flytbase.collect_dock_telemetry(
        access_token="tok",
        org_id="org",
        dock_ids=dock_ids,
        server_region="US",
        window_seconds=1,
    )

    for did in dock_ids:
        assert f"{did}/global_position" in registered_channels
        assert f"{did}/dock_state" in registered_channels
        assert f"{did}/weather" in registered_channels


@pytest.mark.asyncio
async def test_collect_dock_telemetry_captures_dock_location(mocker, sample_dock_global_position):
    registered_events = {}
    registered_channels_map = {}

    mock_sio = AsyncMock()
    mock_sio.connected = True
    mock_sio.connect = AsyncMock()
    mock_sio.disconnect = AsyncMock()

    def fake_event(func):
        registered_events[func.__name__] = func
        return func

    def fake_on(channel, handler):
        registered_channels_map[channel] = handler

    mock_sio.event = fake_event
    mock_sio.on = fake_on

    mocker.patch("app.services.flytbase.socketio.AsyncClient", return_value=mock_sio)

    original_connect = mock_sio.connect

    async def connect_and_fire(*args, **kwargs):
        await original_connect(*args, **kwargs)
        if "connect" in registered_events:
            await registered_events["connect"]()
        # Fire the global_position handler to simulate dock location arrival
        pos_channel = f"{DOCK_ID}/global_position"
        if pos_channel in registered_channels_map:
            await registered_channels_map[pos_channel](sample_dock_global_position)

    mock_sio.connect = connect_and_fire
    mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    result = await flytbase.collect_dock_telemetry(
        access_token="tok",
        org_id="org",
        dock_ids=[DOCK_ID],
        server_region="US",
        window_seconds=1,
    )

    assert result[DOCK_ID]["dock_location"] == (18.5628, 73.7010)


@pytest.mark.asyncio
async def test_collect_dock_telemetry_skips_channels_when_disabled(mocker):
    registered_events = {}
    registered_channels = []
    _make_dock_sio_mock(mocker, registered_events, registered_channels)
    mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    await flytbase.collect_dock_telemetry(
        access_token="tok",
        org_id="org",
        dock_ids=[DOCK_ID],
        server_region="US",
        window_seconds=1,
        collect_dock_state=False,
        collect_dock_weather=False,
    )

    assert f"{DOCK_ID}/global_position" in registered_channels
    assert f"{DOCK_ID}/dock_state" not in registered_channels
    assert f"{DOCK_ID}/weather" not in registered_channels


@pytest.mark.asyncio
async def test_collect_dock_telemetry_returns_empty_when_no_messages(mocker):
    registered_events = {}
    registered_channels = []
    _make_dock_sio_mock(mocker, registered_events, registered_channels)
    mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    result = await flytbase.collect_dock_telemetry(
        access_token="tok",
        org_id="org",
        dock_ids=[DOCK_ID],
        server_region="EU",
        window_seconds=1,
    )

    assert result[DOCK_ID]["dock_state"] == []
    assert result[DOCK_ID]["weather"] == []
    assert result[DOCK_ID]["dock_location"] is None


# ── transform_dock_state_to_observation tests ─────────────────────────────────

def test_transform_dock_state_full(sample_dock_state):
    obs = flytbase.transform_dock_state_to_observation(
        dock_id=DOCK_ID,
        state_data=sample_dock_state,
        recorded_at=RECORDED_AT,
        dock_lat=18.5628,
        dock_lon=73.7010,
        subject_type="dock",
        dock_name="Dock Alpha",
    )

    assert obs is not None
    assert obs["source"] == DOCK_ID
    assert obs["source_name"] == "Dock Alpha"
    assert obs["subject_type"] == "dock"
    assert obs["recorded_at"] == RECORDED_AT
    assert obs["location"] == {"lat": 18.5628, "lon": 73.7010}
    assert obs["additional"]["dock_state"] == 1
    assert obs["additional"]["enclosure_state"] == 0
    assert obs["additional"]["charging_rods_state"] == 1
    assert obs["additional"]["drone_charging_state"] == 1
    assert obs["additional"]["drone_power_state"] == 1
    assert obs["additional"]["emergency_stop_state"] == 0
    assert obs["additional"]["occupancy_state"] == 1
    assert obs["additional"]["connected"] is True
    assert obs["additional"]["operation_mode_state"] == 1
    assert obs["additional"]["total_flight_operations"] == 42


def test_transform_dock_state_default_source_name(sample_dock_state):
    obs = flytbase.transform_dock_state_to_observation(
        dock_id=DOCK_ID,
        state_data=sample_dock_state,
        recorded_at=RECORDED_AT,
        dock_lat=18.5628,
        dock_lon=73.7010,
    )
    assert obs["source_name"] == DOCK_ID


def test_transform_dock_state_returns_none_without_location(sample_dock_state):
    assert flytbase.transform_dock_state_to_observation(
        dock_id=DOCK_ID, state_data=sample_dock_state, recorded_at=RECORDED_AT,
        dock_lat=None, dock_lon=73.7010,
    ) is None
    assert flytbase.transform_dock_state_to_observation(
        dock_id=DOCK_ID, state_data=sample_dock_state, recorded_at=RECORDED_AT,
        dock_lat=18.5628, dock_lon=None,
    ) is None


def test_transform_dock_state_missing_nested_dicts():
    """Minimal payload with no sub-dicts should not raise."""
    obs = flytbase.transform_dock_state_to_observation(
        dock_id=DOCK_ID,
        state_data={"dock_state": 0, "connected": False},
        recorded_at=RECORDED_AT,
        dock_lat=18.5628,
        dock_lon=73.7010,
    )
    assert obs is not None
    assert obs["additional"]["enclosure_state"] is None
    assert obs["additional"]["charging_rods_state"] is None


# ── transform_dock_weather_to_observation tests ───────────────────────────────

def test_transform_dock_weather_full(sample_dock_weather):
    obs = flytbase.transform_dock_weather_to_observation(
        dock_id=DOCK_ID,
        weather_data=sample_dock_weather,
        recorded_at=RECORDED_AT,
        dock_lat=18.5628,
        dock_lon=73.7010,
        subject_type="dock",
        dock_name="Dock Alpha",
    )

    assert obs is not None
    assert obs["source"] == DOCK_ID
    assert obs["source_name"] == "Dock Alpha"
    assert obs["subject_type"] == "dock"
    assert obs["recorded_at"] == RECORDED_AT
    assert obs["location"] == {"lat": 18.5628, "lon": 73.7010}
    assert obs["additional"]["humidity_pct"] == 65.0
    assert obs["additional"]["rainfall"] == 0
    assert obs["additional"]["temperature_c"] == 28.5
    assert obs["additional"]["wind_speed_ms"] == 3.2
    assert obs["additional"]["wind_direction_deg"] == 180


def test_transform_dock_weather_default_source_name(sample_dock_weather):
    obs = flytbase.transform_dock_weather_to_observation(
        dock_id=DOCK_ID,
        weather_data=sample_dock_weather,
        recorded_at=RECORDED_AT,
        dock_lat=18.5628,
        dock_lon=73.7010,
    )
    assert obs["source_name"] == DOCK_ID


def test_transform_dock_weather_returns_none_without_location(sample_dock_weather):
    assert flytbase.transform_dock_weather_to_observation(
        dock_id=DOCK_ID, weather_data=sample_dock_weather, recorded_at=RECORDED_AT,
        dock_lat=None, dock_lon=73.7010,
    ) is None
    assert flytbase.transform_dock_weather_to_observation(
        dock_id=DOCK_ID, weather_data=sample_dock_weather, recorded_at=RECORDED_AT,
        dock_lat=18.5628, dock_lon=None,
    ) is None


def test_transform_dock_weather_missing_nested_dicts():
    """Empty weather payload should not raise."""
    obs = flytbase.transform_dock_weather_to_observation(
        dock_id=DOCK_ID,
        weather_data={},
        recorded_at=RECORDED_AT,
        dock_lat=18.5628,
        dock_lon=73.7010,
    )
    assert obs is not None
    assert obs["additional"]["humidity_pct"] is None
    assert obs["additional"]["wind_speed_ms"] is None
