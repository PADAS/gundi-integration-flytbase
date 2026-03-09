from typing import Dict, List, Optional

import pydantic
from pydantic import validator

from app.actions.core import AuthActionConfiguration, ExecutableActionMixin, PullActionConfiguration
from app.services.utils import FieldWithUIOptions, GlobalUISchemaOptions, UIOptions


class FlytBaseAuthConfig(AuthActionConfiguration, ExecutableActionMixin):
    """
    Credentials for authenticating against the FlytBase OAuth2 endpoint.
    Run this action manually from the Gundi portal to validate credentials and
    store tokens for use by the pull_observations action.
    """

    client_id: str = FieldWithUIOptions(
        ...,
        title="Client ID",
        description="FlytBase OAuth2 Client ID from Settings → Integrations.",
        ui_options=UIOptions(widget="text"),
    )
    client_secret: pydantic.SecretStr = FieldWithUIOptions(
        ...,
        title="Client Secret",
        description="FlytBase OAuth2 Client Secret.",
        ui_options=UIOptions(widget="password"),
    )
    org_id: str = FieldWithUIOptions(
        ...,
        title="Organization ID",
        description="FlytBase Organization ID (MongoDB ObjectId format).",
        ui_options=UIOptions(widget="text"),
    )
    server_region: str = FieldWithUIOptions(
        "US",
        title="Server Region",
        description="FlytBase server region. US: api.flytbase.com, EU: api-eu.flytbase.com",
        ui_options=UIOptions(
            widget="select",
            enumNames=["United States", "European Union"],
        ),
    )

    @validator("server_region")
    def validate_region(cls, v):
        if v.upper() not in ("US", "EU"):
            raise ValueError("server_region must be 'US' or 'EU'")
        return v.upper()

    ui_global_options = GlobalUISchemaOptions(
        order=["client_id", "client_secret", "org_id", "server_region"],
    )


class FlytBasePullObservationsConfig(PullActionConfiguration):
    """
    Configuration for the scheduled pull action that connects to FlytBase Socket.IO,
    collects drone GPS positions for a time window, and sends them to Gundi as Observations.

    Credentials (client_id, client_secret, org_id, server_region) are taken from the
    auth action configuration — run the auth action first to validate credentials.
    """

    drone_ids: List[str] = FieldWithUIOptions(
        ...,
        title="Drone IDs",
        description=(
            "FlytBase drone device IDs to subscribe to (MongoDB ObjectId format). "
            "Find these in the FlytBase dashboard under your drone devices."
        ),
        ui_options=UIOptions(widget="text"),
    )
    window_duration_seconds: int = FieldWithUIOptions(
        270,
        ge=30,
        le=480,
        title="Collection Window (seconds)",
        description=(
            "How long to listen for telemetry per action run. "
            "Default 270s (4.5 min). Maximum 480s. "
            "Should be less than the cron interval to avoid overlap."
        ),
        ui_options=UIOptions(widget="range"),
    )
    subject_type: str = FieldWithUIOptions(
        "drone",
        title="Subject Type",
        description="Gundi subject type tag applied to all observations (e.g. 'drone', 'uav').",
        ui_options=UIOptions(widget="text"),
    )
    drone_name_map: Optional[Dict[str, str]] = FieldWithUIOptions(
        None,
        title="Drone Name Map",
        description=(
            'Optional JSON map of drone_id to human-readable display name. '
            'Example: {"648f2a3d7b1c9e5f4a8d0c2e": "Alpha 1"}. '
            "If omitted, drone_id is used as the source name."
        ),
    )
    dock_ids: Optional[List[str]] = FieldWithUIOptions(
        None,
        title="Dock IDs",
        description=(
            "FlytBase dock device IDs to subscribe to (MongoDB ObjectId format). "
            "Find these in the FlytBase dashboard under your docking station devices. "
            "If omitted, no dock telemetry is collected."
        ),
        ui_options=UIOptions(widget="text"),
    )
    dock_name_map: Optional[Dict[str, str]] = FieldWithUIOptions(
        None,
        title="Dock Name Map",
        description=(
            'Optional JSON map of dock_id to human-readable display name. '
            'Example: {"507f1f77bcf86cd799439022": "Dock Alpha"}. '
            "If omitted, dock_id is used as the source name."
        ),
    )
    dock_subject_type: str = FieldWithUIOptions(
        "dock",
        title="Dock Subject Type",
        description="Gundi subject type tag applied to all dock observations (e.g. 'dock').",
        ui_options=UIOptions(widget="text"),
    )
    collect_dock_state: bool = FieldWithUIOptions(
        True,
        title="Collect Dock State",
        description="Subscribe to {dockId}/dock_state channel (operational state, enclosure, charging).",
        ui_options=UIOptions(widget="checkbox"),
    )
    collect_dock_weather: bool = FieldWithUIOptions(
        True,
        title="Collect Dock Weather",
        description="Subscribe to {dockId}/weather channel (temperature, humidity, wind, rainfall).",
        ui_options=UIOptions(widget="checkbox"),
    )

    ui_global_options = GlobalUISchemaOptions(
        order=[
            "drone_ids", "window_duration_seconds", "subject_type", "drone_name_map",
            "dock_ids", "dock_name_map", "dock_subject_type",
            "collect_dock_state", "collect_dock_weather",
        ],
    )
