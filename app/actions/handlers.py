import asyncio
import logging
from typing import Optional

from app.actions.configurations import FlytBaseAuthConfig, FlytBasePullObservationsConfig
from app.services import flytbase
from app.services.action_scheduler import crontab_schedule
from app.services.activity_logger import activity_logger
from app.services.gundi import send_observations_to_gundi
from app.services.state import IntegrationStateManager
from app.services.utils import generate_batches

logger = logging.getLogger(__name__)

state_manager = IntegrationStateManager()

OBSERVATIONS_BATCH_SIZE = 200


@activity_logger()
async def action_auth(integration, action_config: FlytBaseAuthConfig):
    """
    Validates FlytBase credentials by performing an OAuth2 token exchange.
    Stores access and refresh tokens in the state manager (Redis) for use by
    the pull_observations action. Run this manually from the Gundi portal first.
    """
    integration_id = str(integration.id)
    logger.info(f"Authenticating FlytBase for integration {integration_id}")

    token_response = await flytbase.get_flytbase_token(
        client_id=action_config.client_id,
        client_secret=action_config.client_secret.get_secret_value(),
    )

    await state_manager.set_state(
        integration_id=integration_id,
        action_id="auth",
        state={
            "access_token": token_response["access"]["token"],
            "access_token_expiry": token_response["access"].get("expiry"),
            "refresh_token": token_response["refresh"]["token"],
            "refresh_token_expiry": token_response["refresh"].get("expiry"),
        },
    )

    logger.info(f"FlytBase tokens stored for integration {integration_id}")
    return {"status": "authenticated"}


@activity_logger()
@crontab_schedule("*/5 * * * *")  # Every 5 minutes
async def action_pull_observations(integration, action_config: FlytBasePullObservationsConfig):
    """
    Connects to FlytBase Socket.IO, subscribes to each configured drone's
    global_position channel, collects position messages for window_duration_seconds,
    transforms them into Gundi Observations, and sends them in batches.

    Token refresh decision:
      1. Use cached access token if not expiring within 2 min.
      2. Refresh using refresh token if available and not expired.
      3. Fall back to full re-authentication using auth action credentials.
    """
    integration_id = str(integration.id)
    logger.info(
        f"Starting pull_observations for integration {integration_id}. "
        f"Drones: {action_config.drone_ids}, window: {action_config.window_duration_seconds}s"
    )

    # ── Step 1: Get a valid access token ────────────────────────────────────
    access_token = await _get_valid_access_token(integration)

    # ── Step 2: Retrieve connection settings from auth config ────────────────
    auth_data = _get_auth_config_data(integration)
    if not auth_data:
        raise ValueError(
            "Auth action configuration not found on integration. "
            "Please configure and run the 'auth' action first."
        )
    org_id = auth_data["org_id"]
    server_region = auth_data.get("server_region", "US")

    # ── Step 3: Collect positions via Socket.IO (drone + dock in parallel) ──────
    drone_task = flytbase.collect_drone_positions(
        access_token=access_token,
        org_id=org_id,
        drone_ids=action_config.drone_ids,
        server_region=server_region,
        window_seconds=action_config.window_duration_seconds,
    )
    if action_config.dock_ids:
        dock_task = flytbase.collect_dock_telemetry(
            access_token=access_token,
            org_id=org_id,
            dock_ids=action_config.dock_ids,
            server_region=server_region,
            window_seconds=action_config.window_duration_seconds,
            collect_dock_state=action_config.collect_dock_state,
            collect_dock_weather=action_config.collect_dock_weather,
        )
        collected, dock_collected = await asyncio.gather(drone_task, dock_task)
    else:
        collected = await drone_task
        dock_collected = {}

    # ── Step 4: Transform to Gundi observations ──────────────────────────────
    drone_name_map = action_config.drone_name_map or {}
    observations = []

    for drone_id, messages in collected.items():
        drone_name = drone_name_map.get(drone_id)
        for position_data, recorded_at in messages:
            obs = flytbase.transform_position_to_observation(
                drone_id=drone_id,
                position_data=position_data,
                recorded_at=recorded_at,
                subject_type=action_config.subject_type,
                drone_name=drone_name,
            )
            if obs is not None:
                observations.append(obs)

    dock_name_map = action_config.dock_name_map or {}
    for dock_id, dock_data in dock_collected.items():
        dock_name = dock_name_map.get(dock_id)
        dock_location = dock_data["dock_location"]
        dock_lat, dock_lon = dock_location if dock_location else (None, None)

        if action_config.collect_dock_state:
            for state_data, recorded_at in dock_data["dock_state"]:
                obs = flytbase.transform_dock_state_to_observation(
                    dock_id=dock_id,
                    state_data=state_data,
                    recorded_at=recorded_at,
                    dock_lat=dock_lat,
                    dock_lon=dock_lon,
                    subject_type=action_config.dock_subject_type,
                    dock_name=dock_name,
                )
                if obs is not None:
                    observations.append(obs)

        if action_config.collect_dock_weather:
            for weather_data, recorded_at in dock_data["weather"]:
                obs = flytbase.transform_dock_weather_to_observation(
                    dock_id=dock_id,
                    weather_data=weather_data,
                    recorded_at=recorded_at,
                    dock_lat=dock_lat,
                    dock_lon=dock_lon,
                    subject_type=action_config.dock_subject_type,
                    dock_name=dock_name,
                )
                if obs is not None:
                    observations.append(obs)

    total = len(observations)
    logger.info(f"Transformed {total} observations. Sending to Gundi in batches of {OBSERVATIONS_BATCH_SIZE}.")

    # ── Step 5: Send in batches ──────────────────────────────────────────────
    sent = 0
    for batch in generate_batches(observations, OBSERVATIONS_BATCH_SIZE):
        await send_observations_to_gundi(
            observations=batch,
            integration_id=integration_id,
        )
        sent += len(batch)
        logger.info(f"Sent {sent}/{total} observations to Gundi.")

    return {"observations_extracted": total}


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _get_valid_access_token(integration) -> str:
    """
    Returns a valid FlytBase access token, refreshing or re-authenticating as needed.

    Decision tree:
      - Cached access token valid → return it
      - Cached refresh token valid → refresh → store → return new access token
      - Both expired/missing → full re-auth using auth config credentials → store → return
    """
    integration_id = str(integration.id)
    token_state = await state_manager.get_state(
        integration_id=integration_id,
        action_id="auth",
    )

    access_token = token_state.get("access_token")
    access_expiry = token_state.get("access_token_expiry")
    refresh_token = token_state.get("refresh_token")
    refresh_expiry = token_state.get("refresh_token_expiry")

    if access_token and not flytbase.is_token_expired(access_expiry):
        return access_token

    if refresh_token and not flytbase.is_token_expired(refresh_expiry, buffer_seconds=0):
        logger.info("Access token expiring; refreshing using refresh token.")
        token_response = await flytbase.refresh_flytbase_token(refresh_token)
    else:
        logger.warning(
            "Both access and refresh tokens missing or expired. Re-authenticating."
        )
        auth_data = _get_auth_config_data(integration)
        if not auth_data:
            raise ValueError(
                "Cannot re-authenticate: auth action configuration missing. "
                "Please configure and run the 'auth' action manually."
            )
        token_response = await flytbase.get_flytbase_token(
            client_id=auth_data["client_id"],
            client_secret=auth_data["client_secret"],
        )

    new_state = {
        "access_token": token_response["access"]["token"],
        "access_token_expiry": token_response["access"].get("expiry"),
        "refresh_token": token_response["refresh"]["token"],
        "refresh_token_expiry": token_response["refresh"].get("expiry"),
    }
    await state_manager.set_state(
        integration_id=integration_id,
        action_id="auth",
        state=new_state,
    )
    return new_state["access_token"]


def _get_auth_config_data(integration) -> Optional[dict]:
    """
    Extracts the raw data dict from the 'auth' action configuration on the integration.
    Returns None if the auth configuration is not present.
    """
    for config in integration.configurations:
        if config.action.value == "auth":
            return config.data
    return None
