import json

import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import doosan_policy


@pytest.mark.parametrize(
    ("representation", "state_mode", "expected_dim"),
    [
        ("rotvec_principal", "no_wrench", 19),
        ("rotvec_principal", "full", 25),
        ("rotvec_continuous", "no_wrench", 19),
        ("rotvec_continuous", "full", 25),
        ("quaternion", "no_wrench", 20),
        ("quaternion", "full", 26),
        ("rotation6d", "no_wrench", 22),
        ("rotation6d", "full", 28),
    ],
)
def test_expected_state_dim(representation, state_mode, expected_dim):
    assert doosan_policy.expected_state_dim(representation, state_mode) == expected_dim


def test_inputs_map_cameras_and_preserve_actions_exactly():
    transform = doosan_policy.DoosanInputs(
        model_type=_model.ModelType.PI05,
        orientation_representation="rotation6d",
        state_mode="full",
    )
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)
    data = {
        "observation/external_camera_2": np.ones((3, 480, 848), dtype=np.float32),
        "observation/tcp_camera": np.ones((3, 480, 640), dtype=np.float32),
        "observation/state": np.arange(28, dtype=np.float32),
        "actions": actions.copy(),
        "prompt": b"insert the peg",
    }

    result = transform(data)

    assert result["state"].shape == (28,)
    assert result["image"]["base_0_rgb"].shape == (480, 848, 3)
    assert result["image"]["left_wrist_0_rgb"].shape == (480, 640, 3)
    assert result["image"]["right_wrist_0_rgb"].shape == (480, 640, 3)
    assert result["image"]["base_0_rgb"].dtype == np.uint8
    assert result["image_mask"]["base_0_rgb"]
    assert result["image_mask"]["left_wrist_0_rgb"]
    assert not result["image_mask"]["right_wrist_0_rgb"]
    np.testing.assert_array_equal(result["actions"], actions)
    assert result["prompt"] == "insert the peg"


def test_inputs_reject_wrong_state_width_even_when_rotvec_profiles_share_width():
    transform = doosan_policy.DoosanInputs(
        model_type=_model.ModelType.PI05,
        orientation_representation="quaternion",
        state_mode="full",
    )
    data = doosan_policy.make_doosan_example(
        orientation_representation="rotvec_principal",
        state_mode="full",
    )

    with pytest.raises(ValueError, match="width mismatch"):
        transform(data)


def test_inputs_reject_non_pi05_model_type():
    transform = doosan_policy.DoosanInputs(model_type=_model.ModelType.PI0)
    with pytest.raises(ValueError, match="requires PI05"):
        transform(doosan_policy.make_doosan_example())


def test_inputs_reject_nonfinite_state_and_actions():
    transform = doosan_policy.DoosanInputs(model_type=_model.ModelType.PI05)
    data = doosan_policy.make_doosan_example()
    data["observation/state"][0] = np.nan
    with pytest.raises(ValueError, match="state contains non-finite"):
        transform(data)

    data = doosan_policy.make_doosan_example()
    data["actions"] = np.zeros((2, 7), dtype=np.float32)
    data["actions"][0, 0] = np.inf
    with pytest.raises(ValueError, match="actions contain non-finite"):
        transform(data)


def test_model_state_profile_metadata_is_fail_closed():
    profile = {
        "schema_version": "doosan_model_state_profile_v1",
        "profile_id": "doosan_full_rotvec_continuous_v1",
        "orientation_representation": "rotvec_continuous",
        "include_wrench": True,
        "state_dim": 25,
        "state_mode": "full",
        "wrench_policy": "final_six_channels",
    }
    doosan_policy.validate_model_state_profile_metadata(
        profile,
        orientation_representation="rotvec_continuous",
        state_mode="full",
    )

    with pytest.raises(ValueError, match="metadata mismatch"):
        doosan_policy.validate_model_state_profile_metadata(
            profile,
            orientation_representation="rotvec_principal",
            state_mode="full",
        )


def test_outputs_return_only_seven_semantic_action_dimensions():
    actions = np.arange(3 * 32, dtype=np.float32).reshape(3, 32)
    result = doosan_policy.DoosanOutputs()({"actions": actions})
    assert result["actions"].shape == (3, 7)
    np.testing.assert_array_equal(result["actions"], actions[:, :7])


def _write_export_provenance(tmp_path, *, profile=None, state_dim=25):
    meta = tmp_path / "meta"
    meta.mkdir(parents=True)
    payload = {
        "schema_version": "doosan_forcevla_lerobot_v21_export_v1",
        "state_dim": state_dim,
        "action_dim": 7,
    }
    if profile is not None:
        payload["model_state_profile"] = profile
    (meta / "export_provenance.json").write_text(json.dumps(payload))


def test_lerobot_export_provenance_accepts_explicit_profile(tmp_path):
    profile = {
        "schema_version": "doosan_model_state_profile_v1",
        "profile_id": "doosan_full_rotation6d_v1",
        "orientation_representation": "rotation6d",
        "include_wrench": True,
        "state_dim": 28,
        "state_mode": "full",
        "wrench_policy": "final_six_channels",
    }
    _write_export_provenance(tmp_path, profile=profile, state_dim=28)

    doosan_policy.validate_lerobot_export_provenance(
        tmp_path,
        orientation_representation="rotation6d",
        state_mode="full",
    )


def test_lerobot_export_provenance_rejects_missing_profile_by_default(tmp_path):
    _write_export_provenance(tmp_path, state_dim=25)

    with pytest.raises(ValueError, match="missing model_state_profile"):
        doosan_policy.validate_lerobot_export_provenance(
            tmp_path,
            orientation_representation="rotvec_principal",
            state_mode="full",
        )


def test_lerobot_export_provenance_legacy_default_requires_explicit_opt_out(tmp_path):
    _write_export_provenance(tmp_path, state_dim=25)

    doosan_policy.validate_lerobot_export_provenance(
        tmp_path,
        orientation_representation="rotvec_principal",
        state_mode="full",
        require_explicit_profile=False,
    )


def test_lerobot_export_provenance_rejects_equal_width_wrong_rotvec_profile(tmp_path):
    profile = {
        "schema_version": "doosan_model_state_profile_v1",
        "profile_id": "doosan_full_rotvec_continuous_v1",
        "orientation_representation": "rotvec_continuous",
        "include_wrench": True,
        "state_dim": 25,
        "state_mode": "full",
        "wrench_policy": "final_six_channels",
    }
    _write_export_provenance(tmp_path, profile=profile, state_dim=25)

    with pytest.raises(ValueError, match="metadata mismatch"):
        doosan_policy.validate_lerobot_export_provenance(
            tmp_path,
            orientation_representation="rotvec_principal",
            state_mode="full",
        )
