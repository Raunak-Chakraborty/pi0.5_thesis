"""Doosan M1013 policy transforms for the thesis π0.5 adapter.

The data-conversion repository owns physical synchronization and representation
selection. This module only maps an already-exported Doosan LeRobot sample into
OpenPI's model-facing dictionary and validates the selected state profile.

Important action contract: Doosan actions are already 7D semantic pose deltas
``[dxyz, spatial/base-frame relative rotvec, absolute gripper target]``. They
must not pass through :class:`openpi.transforms.DeltaActions` again.
"""

from collections.abc import Mapping
import dataclasses
from typing import Literal, TypeAlias

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

OrientationRepresentation: TypeAlias = Literal[
    "rotvec_principal",
    "rotvec_continuous",
    "quaternion",
    "rotation6d",
]
StateMode: TypeAlias = Literal["no_wrench", "full"]

MODEL_STATE_PROFILE_SCHEMA_ID = "doosan_model_state_profile_v1"
SEMANTIC_ACTION_DIM = 7

_STATE_DIMS: dict[tuple[str, str], int] = {
    ("rotvec_principal", "no_wrench"): 19,
    ("rotvec_principal", "full"): 25,
    ("rotvec_continuous", "no_wrench"): 19,
    ("rotvec_continuous", "full"): 25,
    ("quaternion", "no_wrench"): 20,
    ("quaternion", "full"): 26,
    ("rotation6d", "no_wrench"): 22,
    ("rotation6d", "full"): 28,
}


def expected_state_dim(
    orientation_representation: OrientationRepresentation,
    state_mode: StateMode,
) -> int:
    """Return the exact exported Doosan observation-state width."""
    try:
        return _STATE_DIMS[(orientation_representation, state_mode)]
    except KeyError as exc:
        raise ValueError(
            "unsupported Doosan model-state profile: "
            f"orientation_representation={orientation_representation!r}, state_mode={state_mode!r}"
        ) from exc


def validate_model_state_profile_metadata(
    profile: Mapping[str, object],
    *,
    orientation_representation: OrientationRepresentation,
    state_mode: StateMode,
) -> None:
    """Fail closed when exporter provenance disagrees with the selected profile.

    Non-default data-tools exports write this object under
    ``meta/export_provenance.json -> model_state_profile``. The legacy default
    export may omit it; callers that require representation-explicit provenance
    should require the object before invoking this validator.
    """
    expected_dim = expected_state_dim(orientation_representation, state_mode)
    expected_wrench = state_mode == "full"
    expected_wrench_policy = "final_six_channels" if expected_wrench else "omitted"

    expected = {
        "schema_version": MODEL_STATE_PROFILE_SCHEMA_ID,
        "orientation_representation": orientation_representation,
        "include_wrench": expected_wrench,
        "state_dim": expected_dim,
        "wrench_policy": expected_wrench_policy,
    }
    mismatches = {
        key: (profile.get(key), value)
        for key, value in expected.items()
        if profile.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: got {actual!r}, expected {wanted!r}"
            for key, (actual, wanted) in mismatches.items()
        )
        raise ValueError(f"Doosan model_state_profile metadata mismatch: {details}")


def make_doosan_example(
    *,
    orientation_representation: OrientationRepresentation = "rotvec_principal",
    state_mode: StateMode = "full",
) -> dict:
    """Create one synthetic Doosan inference input."""
    return {
        "observation/external_camera_2": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
        "observation/tcp_camera": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/state": np.random.rand(expected_state_dim(orientation_representation, state_mode)),
        "prompt": "insert the peg into the hole",
    }


def _parse_image(image: object) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Doosan image must be rank 3, got shape {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Doosan image must have 3 RGB channels, got shape {image.shape}")
    return image


@dataclasses.dataclass(frozen=True)
class DoosanInputs(transforms.DataTransformFn):
    """Map exported Doosan data to π0.5's common observation/action format."""

    model_type: _model.ModelType
    orientation_representation: OrientationRepresentation = "rotvec_principal"
    state_mode: StateMode = "full"

    def __call__(self, data: dict) -> dict:
        if self.model_type != _model.ModelType.PI05:
            raise ValueError(f"Doosan thesis adapter requires PI05, got {self.model_type}")

        state = np.asarray(data["observation/state"])
        expected_dim = expected_state_dim(self.orientation_representation, self.state_mode)
        if state.ndim != 1 or state.shape[0] != expected_dim:
            raise ValueError(
                "Doosan observation.state width mismatch: "
                f"got shape {state.shape}, expected ({expected_dim},) for "
                f"{self.orientation_representation}/{self.state_mode}"
            )
        if not np.all(np.isfinite(state)):
            raise ValueError("Doosan observation.state contains non-finite values")

        base_image = _parse_image(data["observation/external_camera_2"])
        wrist_image = _parse_image(data["observation/tcp_camera"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(wrist_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.ndim < 1 or actions.shape[-1] != SEMANTIC_ACTION_DIM:
                raise ValueError(
                    "Doosan actions must have final dimension 7, "
                    f"got shape {actions.shape}"
                )
            if not np.all(np.isfinite(actions)):
                raise ValueError("Doosan actions contain non-finite values")
            # Preserve the converter-owned semantic action exactly. In particular,
            # do not apply DeltaActions here: the first six values are already
            # Cartesian/rotation deltas and the final gripper target is absolute.
            inputs["actions"] = actions

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class DoosanOutputs(transforms.DataTransformFn):
    """Strip OpenPI's internal action padding from π0.5 inference output."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.shape[-1] < SEMANTIC_ACTION_DIM:
            raise ValueError(
                f"π0.5 output action width must be at least 7, got shape {actions.shape}"
            )
        return {"actions": actions[..., :SEMANTIC_ACTION_DIM]}
