"""
Tests for FlytBase action handlers (action_auth, action_pull_observations).
"""
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from gundi_core.schemas.v2 import Integration

from app.actions.configurations import FlytBaseAuthConfig, FlytBasePullObservationsConfig
from app.actions.handlers import action_auth, action_pull_observations


# ── Fixtures ──────────────────────────────────────────────────────────────────

INTEGRATION_ID = "779ff3ab-5589-4f4c-9e0a-ae8d6c9edff0"
DRONE_ID = "648f2a3d7b1c9e5f4a8d0c2e"


@pytest.fixture
def mock_token_response_fresh():
    future = (datetime.now(timezone.utc) + timedelta(minutes=14)).isoformat()
    refresh_future = (datetime.now(timezone.utc) + timedelta(days=6)).isoformat()
    return {
        "access": {"token": "fresh-access-token", "expiry": future},
        "refresh": {"token": "fresh-refresh-token", "expiry": refresh_future},
    }


@pytest.fixture
def valid_token_state():
    """Token state with a non-expiring access token."""
    future = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    refresh_future = (datetime.now(timezone.utc) + timedelta(days=6)).isoformat()
    return {
        "access_token": "cached-access-token",
        "access_token_expiry": future,
        "refresh_token": "cached-refresh-token",
        "refresh_token_expiry": refresh_future,
    }


@pytest.fixture
def expired_access_token_state():
    """Token state where access token is expired but refresh token is valid."""
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    refresh_future = (datetime.now(timezone.utc) + timedelta(days=6)).isoformat()
    return {
        "access_token": "expired-access-token",
        "access_token_expiry": past,
        "refresh_token": "valid-refresh-token",
        "refresh_token_expiry": refresh_future,
    }


@pytest.fixture
def both_tokens_expired_state():
    """Token state where both access and refresh tokens are expired."""
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    return {
        "access_token": "expired-access-token",
        "access_token_expiry": past,
        "refresh_token": "expired-refresh-token",
        "refresh_token_expiry": past,
    }


@pytest.fixture
def flytbase_integration(integration_v2_as_dict):
    """
    Integration object with a FlytBase auth config embedded in configurations.
    Replaces the existing auth config data with FlytBase-specific fields.
    """
    data = dict(integration_v2_as_dict)
    # Replace auth config data with FlytBase credentials
    for config in data["configurations"]:
        if config["action"]["value"] == "auth":
            config["data"] = {
                "client_id": "test-client-id",
                "client_secret": "test-client-secret",
                "org_id": "test-org-id",
                "server_region": "US",
            }
    return Integration.parse_obj(data)


@pytest.fixture
def auth_config():
    return FlytBaseAuthConfig(
        client_id="test-client-id",
        client_secret="test-client-secret",
        org_id="test-org-id",
        server_region="US",
    )


@pytest.fixture
def pull_config():
    return FlytBasePullObservationsConfig(
        drone_ids=[DRONE_ID],
        window_duration_seconds=30,
        subject_type="drone",
    )


@pytest.fixture
def pull_config_with_names():
    return FlytBasePullObservationsConfig(
        drone_ids=[DRONE_ID],
        window_duration_seconds=30,
        subject_type="drone",
        drone_name_map={DRONE_ID: "Alpha 1"},
    )


@pytest.fixture
def sample_positions():
    """Two valid position tuples (data, recorded_at)."""
    ts = datetime.now(timezone.utc).isoformat()
    pos = {
        "position": {"latitude": 18.5664, "longitude": 73.7719, "elevation": 550.0, "height": 25.0, "gps_satellites": 12},
        "speed": {"horizontal": 4.0, "vertical": 0.5},
        "home_position": {"latitude": 18.566, "longitude": 73.771, "distance": 50.0},
        "rtk": {"quality": 2, "rtk_satellites": 8, "fix_state": 2},
    }
    return [(pos, ts), (pos, ts)]


@pytest.fixture
def mock_state_manager(mocker, valid_token_state):
    sm = MagicMock()
    sm.get_state = AsyncMock(return_value=valid_token_state)
    sm.set_state = AsyncMock(return_value=None)
    mocker.patch("app.actions.handlers.state_manager", sm)
    return sm


# ── action_auth tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_action_auth_stores_tokens(
    mocker, flytbase_integration, auth_config, mock_token_response_fresh, mock_publish_event
):
    """action_auth should call get_flytbase_token and store tokens in state manager."""
    mock_get_token = AsyncMock(return_value=mock_token_response_fresh)
    mocker.patch("app.services.flytbase.get_flytbase_token", mock_get_token)
    mocker.patch("app.services.activity_logger.publish_event", mock_publish_event)

    mock_sm = MagicMock()
    mock_sm.set_state = AsyncMock(return_value=None)
    mocker.patch("app.actions.handlers.state_manager", mock_sm)

    result = await action_auth(integration=flytbase_integration, action_config=auth_config)

    assert result == {"status": "authenticated"}
    mock_get_token.assert_called_once_with(
        client_id="test-client-id",
        client_secret="test-client-secret",
    )
    mock_sm.set_state.assert_called_once()
    call_kwargs = mock_sm.set_state.call_args.kwargs
    assert call_kwargs["action_id"] == "auth"
    assert call_kwargs["state"]["access_token"] == "fresh-access-token"
    assert call_kwargs["state"]["refresh_token"] == "fresh-refresh-token"


@pytest.mark.asyncio
async def test_action_auth_propagates_http_error(
    mocker, flytbase_integration, auth_config, mock_publish_event
):
    """action_auth should propagate HTTP errors from get_flytbase_token."""
    import httpx

    mocker.patch(
        "app.services.flytbase.get_flytbase_token",
        AsyncMock(side_effect=httpx.HTTPStatusError("401", request=MagicMock(), response=MagicMock(status_code=401))),
    )
    mocker.patch("app.actions.handlers.state_manager", MagicMock(set_state=AsyncMock()))
    mocker.patch("app.services.activity_logger.publish_event", mock_publish_event)

    with pytest.raises(httpx.HTTPStatusError):
        await action_auth(integration=flytbase_integration, action_config=auth_config)


# ── action_pull_observations — token handling ─────────────────────────────────

@pytest.mark.asyncio
async def test_pull_uses_cached_valid_token(
    mocker, flytbase_integration, pull_config, mock_publish_event, valid_token_state, sample_positions
):
    """A valid cached access token should be used without calling get/refresh token."""
    mock_sm = MagicMock()
    mock_sm.get_state = AsyncMock(return_value=valid_token_state)
    mock_sm.set_state = AsyncMock(return_value=None)
    mocker.patch("app.actions.handlers.state_manager", mock_sm)

    mock_collect = AsyncMock(return_value={DRONE_ID: sample_positions})
    mocker.patch("app.services.flytbase.collect_drone_positions", mock_collect)
    mocker.patch("app.actions.handlers.send_observations_to_gundi", AsyncMock(return_value=[{}, {}]))
    mocker.patch("app.services.gundi._get_gundi_api_key", AsyncMock(return_value="fake-key"))
    mocker.patch("app.services.activity_logger.publish_event", mock_publish_event)

    mock_get_token = AsyncMock()
    mock_refresh_token = AsyncMock()
    mocker.patch("app.services.flytbase.get_flytbase_token", mock_get_token)
    mocker.patch("app.services.flytbase.refresh_flytbase_token", mock_refresh_token)

    result = await action_pull_observations(
        integration=flytbase_integration, action_config=pull_config
    )

    mock_get_token.assert_not_called()
    mock_refresh_token.assert_not_called()
    mock_collect.assert_called_once()
    assert mock_collect.call_args.kwargs["access_token"] == "cached-access-token"
    assert result["observations_extracted"] == 2


@pytest.mark.asyncio
async def test_pull_refreshes_expired_access_token(
    mocker, flytbase_integration, pull_config, mock_publish_event,
    expired_access_token_state, mock_token_response_fresh, sample_positions
):
    """When access token is expired but refresh token is valid, use refresh."""
    mock_sm = MagicMock()
    mock_sm.get_state = AsyncMock(return_value=expired_access_token_state)
    mock_sm.set_state = AsyncMock(return_value=None)
    mocker.patch("app.actions.handlers.state_manager", mock_sm)

    mock_refresh = AsyncMock(return_value=mock_token_response_fresh)
    mocker.patch("app.services.flytbase.refresh_flytbase_token", mock_refresh)

    mock_get_token = AsyncMock()
    mocker.patch("app.services.flytbase.get_flytbase_token", mock_get_token)

    mock_collect = AsyncMock(return_value={DRONE_ID: sample_positions})
    mocker.patch("app.services.flytbase.collect_drone_positions", mock_collect)
    mocker.patch("app.actions.handlers.send_observations_to_gundi", AsyncMock(return_value=[{}, {}]))
    mocker.patch("app.services.gundi._get_gundi_api_key", AsyncMock(return_value="fake-key"))
    mocker.patch("app.services.activity_logger.publish_event", mock_publish_event)

    await action_pull_observations(integration=flytbase_integration, action_config=pull_config)

    mock_refresh.assert_called_once_with("valid-refresh-token")
    mock_get_token.assert_not_called()
    # New token stored
    mock_sm.set_state.assert_called()
    stored = mock_sm.set_state.call_args.kwargs["state"]
    assert stored["access_token"] == "fresh-access-token"


@pytest.mark.asyncio
async def test_pull_full_reauth_when_both_tokens_expired(
    mocker, flytbase_integration, pull_config, mock_publish_event,
    both_tokens_expired_state, mock_token_response_fresh, sample_positions
):
    """When both tokens are expired, fall back to full re-authentication."""
    mock_sm = MagicMock()
    mock_sm.get_state = AsyncMock(return_value=both_tokens_expired_state)
    mock_sm.set_state = AsyncMock(return_value=None)
    mocker.patch("app.actions.handlers.state_manager", mock_sm)

    mock_get_token = AsyncMock(return_value=mock_token_response_fresh)
    mocker.patch("app.services.flytbase.get_flytbase_token", mock_get_token)

    mock_refresh = AsyncMock()
    mocker.patch("app.services.flytbase.refresh_flytbase_token", mock_refresh)

    mock_collect = AsyncMock(return_value={DRONE_ID: sample_positions})
    mocker.patch("app.services.flytbase.collect_drone_positions", mock_collect)
    mocker.patch("app.actions.handlers.send_observations_to_gundi", AsyncMock(return_value=[{}, {}]))
    mocker.patch("app.services.gundi._get_gundi_api_key", AsyncMock(return_value="fake-key"))
    mocker.patch("app.services.activity_logger.publish_event", mock_publish_event)

    await action_pull_observations(integration=flytbase_integration, action_config=pull_config)

    mock_get_token.assert_called_once_with(
        client_id="test-client-id",
        client_secret="test-client-secret",
    )
    mock_refresh.assert_not_called()


# ── action_pull_observations — data flow ─────────────────────────────────────

@pytest.mark.asyncio
async def test_pull_sends_observations_to_gundi(
    mocker, flytbase_integration, pull_config, mock_publish_event,
    valid_token_state, sample_positions
):
    """Observations should be transformed and sent to Gundi."""
    mock_sm = MagicMock()
    mock_sm.get_state = AsyncMock(return_value=valid_token_state)
    mock_sm.set_state = AsyncMock(return_value=None)
    mocker.patch("app.actions.handlers.state_manager", mock_sm)

    mock_collect = AsyncMock(return_value={DRONE_ID: sample_positions})
    mocker.patch("app.services.flytbase.collect_drone_positions", mock_collect)

    mock_send = AsyncMock(return_value=[{}, {}])
    mocker.patch("app.actions.handlers.send_observations_to_gundi", mock_send)
    mocker.patch("app.services.gundi._get_gundi_api_key", AsyncMock(return_value="fake-key"))
    mocker.patch("app.services.activity_logger.publish_event", mock_publish_event)

    result = await action_pull_observations(
        integration=flytbase_integration, action_config=pull_config
    )

    assert result == {"observations_extracted": 2}
    mock_send.assert_called_once()
    sent_observations = mock_send.call_args.kwargs["observations"]
    assert len(sent_observations) == 2
    assert sent_observations[0]["source"] == DRONE_ID
    assert sent_observations[0]["subject_type"] == "drone"
    assert "lat" in sent_observations[0]["location"]
    assert "lon" in sent_observations[0]["location"]


@pytest.mark.asyncio
async def test_pull_applies_drone_name_map(
    mocker, flytbase_integration, pull_config_with_names, mock_publish_event,
    valid_token_state, sample_positions
):
    """Drone name map should be applied to source_name in observations."""
    mock_sm = MagicMock()
    mock_sm.get_state = AsyncMock(return_value=valid_token_state)
    mock_sm.set_state = AsyncMock(return_value=None)
    mocker.patch("app.actions.handlers.state_manager", mock_sm)

    mock_collect = AsyncMock(return_value={DRONE_ID: sample_positions})
    mocker.patch("app.services.flytbase.collect_drone_positions", mock_collect)

    mock_send = AsyncMock(return_value=[{}, {}])
    mocker.patch("app.actions.handlers.send_observations_to_gundi", mock_send)
    mocker.patch("app.services.gundi._get_gundi_api_key", AsyncMock(return_value="fake-key"))
    mocker.patch("app.services.activity_logger.publish_event", mock_publish_event)

    await action_pull_observations(
        integration=flytbase_integration, action_config=pull_config_with_names
    )

    sent = mock_send.call_args.kwargs["observations"]
    assert sent[0]["source_name"] == "Alpha 1"


@pytest.mark.asyncio
async def test_pull_filters_missing_coords(
    mocker, flytbase_integration, pull_config, mock_publish_event, valid_token_state
):
    """Positions without lat/lon should be dropped silently."""
    mock_sm = MagicMock()
    mock_sm.get_state = AsyncMock(return_value=valid_token_state)
    mock_sm.set_state = AsyncMock(return_value=None)
    mocker.patch("app.actions.handlers.state_manager", mock_sm)

    no_coords_pos = {
        "position": {"latitude": None, "longitude": None},
        "speed": {},
        "home_position": {},
        "rtk": {},
    }
    ts = datetime.now(timezone.utc).isoformat()
    mock_collect = AsyncMock(return_value={DRONE_ID: [(no_coords_pos, ts), (no_coords_pos, ts)]})
    mocker.patch("app.services.flytbase.collect_drone_positions", mock_collect)

    mock_send = AsyncMock()
    mocker.patch("app.actions.handlers.send_observations_to_gundi", mock_send)
    mocker.patch("app.services.gundi._get_gundi_api_key", AsyncMock(return_value="fake-key"))
    mocker.patch("app.services.activity_logger.publish_event", mock_publish_event)

    result = await action_pull_observations(
        integration=flytbase_integration, action_config=pull_config
    )

    assert result == {"observations_extracted": 0}
    mock_send.assert_not_called()


@pytest.mark.asyncio
async def test_pull_empty_window_no_crash(
    mocker, flytbase_integration, pull_config, mock_publish_event, valid_token_state
):
    """An empty collection window (no drone messages) should return 0 without crashing."""
    mock_sm = MagicMock()
    mock_sm.get_state = AsyncMock(return_value=valid_token_state)
    mock_sm.set_state = AsyncMock(return_value=None)
    mocker.patch("app.actions.handlers.state_manager", mock_sm)

    mock_collect = AsyncMock(return_value={DRONE_ID: []})
    mocker.patch("app.services.flytbase.collect_drone_positions", mock_collect)

    mock_send = AsyncMock()
    mocker.patch("app.actions.handlers.send_observations_to_gundi", mock_send)
    mocker.patch("app.services.gundi._get_gundi_api_key", AsyncMock(return_value="fake-key"))
    mocker.patch("app.services.activity_logger.publish_event", mock_publish_event)

    result = await action_pull_observations(
        integration=flytbase_integration, action_config=pull_config
    )

    assert result == {"observations_extracted": 0}
    mock_send.assert_not_called()


# ── action_pull_observations — dock telemetry ─────────────────────────────────

DOCK_ID = "507f1f77bcf86cd799439022"


@pytest.fixture
def pull_config_with_docks():
    return FlytBasePullObservationsConfig(
        drone_ids=[DRONE_ID],
        window_duration_seconds=30,
        subject_type="drone",
        dock_ids=[DOCK_ID],
        dock_name_map={DOCK_ID: "Dock Alpha"},
        dock_subject_type="dock",
        collect_dock_state=True,
        collect_dock_weather=True,
    )


@pytest.fixture
def sample_dock_telemetry():
    ts = datetime.now(timezone.utc).isoformat()
    state_data = {
        "dock_state": 1, "enclosure": {"state": 0}, "charging_rods": {"state": 1},
        "drone_charging": {"state": 1}, "drone_power": {"state": 1},
        "emergency_stop": {"state": 0}, "occupancy": {"state": 1},
        "operation_mode": {"state": 1}, "connected": True, "total_flight_operations": 10,
    }
    weather_data = {
        "weather": {"humidity": 60.0, "rainfall": 0, "temperature": 25.0,
                    "wind": {"direction": 90, "speed": 2.5}}
    }
    return {
        DOCK_ID: {
            "dock_state": [(state_data, ts), (state_data, ts)],
            "weather": [(weather_data, ts)],
            "dock_location": (18.5628, 73.7010),
        }
    }


def _setup_pull_mocks(mocker, mock_publish_event, valid_token_state,
                      drone_collect_return, dock_collect_return=None):
    mock_sm = MagicMock()
    mock_sm.get_state = AsyncMock(return_value=valid_token_state)
    mock_sm.set_state = AsyncMock(return_value=None)
    mocker.patch("app.actions.handlers.state_manager", mock_sm)

    mock_drone_collect = AsyncMock(return_value=drone_collect_return)
    mocker.patch("app.services.flytbase.collect_drone_positions", mock_drone_collect)

    mock_dock_collect = AsyncMock(return_value=dock_collect_return or {})
    mocker.patch("app.services.flytbase.collect_dock_telemetry", mock_dock_collect)

    mock_send = AsyncMock(return_value=[])
    mocker.patch("app.actions.handlers.send_observations_to_gundi", mock_send)
    mocker.patch("app.services.gundi._get_gundi_api_key", AsyncMock(return_value="fake-key"))
    mocker.patch("app.services.activity_logger.publish_event", mock_publish_event)

    return mock_drone_collect, mock_dock_collect, mock_send


@pytest.mark.asyncio
async def test_pull_skips_dock_collection_when_no_dock_ids(
    mocker, flytbase_integration, pull_config, mock_publish_event,
    valid_token_state, sample_positions
):
    """collect_dock_telemetry should NOT be called when dock_ids is not set."""
    mock_drone_collect, mock_dock_collect, _ = _setup_pull_mocks(
        mocker, mock_publish_event, valid_token_state,
        drone_collect_return={DRONE_ID: sample_positions},
    )

    await action_pull_observations(integration=flytbase_integration, action_config=pull_config)

    mock_dock_collect.assert_not_called()
    mock_drone_collect.assert_called_once()


@pytest.mark.asyncio
async def test_pull_collects_dock_telemetry_when_dock_ids_configured(
    mocker, flytbase_integration, pull_config_with_docks, mock_publish_event,
    valid_token_state, sample_positions, sample_dock_telemetry
):
    """collect_dock_telemetry should be called when dock_ids is configured."""
    mock_drone_collect, mock_dock_collect, _ = _setup_pull_mocks(
        mocker, mock_publish_event, valid_token_state,
        drone_collect_return={DRONE_ID: sample_positions},
        dock_collect_return=sample_dock_telemetry,
    )

    await action_pull_observations(
        integration=flytbase_integration, action_config=pull_config_with_docks
    )

    mock_dock_collect.assert_called_once()
    call_kwargs = mock_dock_collect.call_args.kwargs
    assert call_kwargs["dock_ids"] == [DOCK_ID]
    assert call_kwargs["collect_dock_state"] is True
    assert call_kwargs["collect_dock_weather"] is True


@pytest.mark.asyncio
async def test_pull_sends_dock_state_observations(
    mocker, flytbase_integration, pull_config_with_docks, mock_publish_event,
    valid_token_state, sample_dock_telemetry
):
    """Dock state messages should be transformed and included in the Gundi send."""
    _, _, mock_send = _setup_pull_mocks(
        mocker, mock_publish_event, valid_token_state,
        drone_collect_return={DRONE_ID: []},
        dock_collect_return=sample_dock_telemetry,
    )

    result = await action_pull_observations(
        integration=flytbase_integration, action_config=pull_config_with_docks
    )

    # 2 dock_state + 1 weather = 3 dock observations
    assert result["observations_extracted"] == 3
    mock_send.assert_called_once()
    sent = mock_send.call_args.kwargs["observations"]
    dock_state_obs = [o for o in sent if "dock_state" in o.get("additional", {})]
    assert len(dock_state_obs) == 2
    assert dock_state_obs[0]["source"] == DOCK_ID
    assert dock_state_obs[0]["source_name"] == "Dock Alpha"
    assert dock_state_obs[0]["subject_type"] == "dock"
    assert dock_state_obs[0]["location"] == {"lat": 18.5628, "lon": 73.7010}
    assert dock_state_obs[0]["additional"]["dock_state"] == 1


@pytest.mark.asyncio
async def test_pull_sends_dock_weather_observations(
    mocker, flytbase_integration, pull_config_with_docks, mock_publish_event,
    valid_token_state, sample_dock_telemetry
):
    """Dock weather messages should be transformed and included in the Gundi send."""
    _, _, mock_send = _setup_pull_mocks(
        mocker, mock_publish_event, valid_token_state,
        drone_collect_return={DRONE_ID: []},
        dock_collect_return=sample_dock_telemetry,
    )

    await action_pull_observations(
        integration=flytbase_integration, action_config=pull_config_with_docks
    )

    sent = mock_send.call_args.kwargs["observations"]
    weather_obs = [o for o in sent if "humidity_pct" in o.get("additional", {})]
    assert len(weather_obs) == 1
    assert weather_obs[0]["additional"]["temperature_c"] == 25.0
    assert weather_obs[0]["additional"]["wind_speed_ms"] == 2.5
    assert weather_obs[0]["additional"]["wind_direction_deg"] == 90


@pytest.mark.asyncio
async def test_pull_skips_dock_obs_without_location(
    mocker, flytbase_integration, pull_config_with_docks, mock_publish_event,
    valid_token_state
):
    """Dock observations should be skipped when no dock GPS location was received."""
    ts = datetime.now(timezone.utc).isoformat()
    state_data = {"dock_state": 0, "connected": True}
    weather_data = {"weather": {"humidity": 50.0, "rainfall": 0, "temperature": 20.0,
                                "wind": {"direction": 0, "speed": 1.0}}}
    no_location_dock = {
        DOCK_ID: {
            "dock_state": [(state_data, ts)],
            "weather": [(weather_data, ts)],
            "dock_location": None,
        }
    }

    _, _, mock_send = _setup_pull_mocks(
        mocker, mock_publish_event, valid_token_state,
        drone_collect_return={DRONE_ID: []},
        dock_collect_return=no_location_dock,
    )

    result = await action_pull_observations(
        integration=flytbase_integration, action_config=pull_config_with_docks
    )

    assert result["observations_extracted"] == 0
    mock_send.assert_not_called()
