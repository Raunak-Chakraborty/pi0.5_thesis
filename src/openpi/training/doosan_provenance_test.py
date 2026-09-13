import json

import pytest

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import data_loader


def _write_provenance(root, *, representation="rotation6d", state_mode="full", state_dim=28):
    meta = root / "meta"
    meta.mkdir(parents=True)
    profile = {
        "schema_version": "doosan_model_state_profile_v1",
        "profile_id": f"doosan_{state_mode}_{representation}_v1",
        "orientation_representation": representation,
        "include_wrench": state_mode == "full",
        "state_dim": state_dim,
        "state_mode": state_mode,
        "wrench_policy": "final_six_channels" if state_mode == "full" else "omitted",
    }
    payload = {
        "schema_version": "doosan_forcevla_lerobot_v21_export_v1",
        "state_dim": state_dim,
        "action_dim": 7,
        "model_state_profile": profile,
    }
    (meta / "export_provenance.json").write_text(json.dumps(payload))


def _patch_lerobot(monkeypatch, root):
    class Metadata:
        def __init__(self, repo_id):
            self.root = root
            self.fps = 30
            self.tasks = {}

    sentinel = object()
    monkeypatch.setattr(data_loader.lerobot_dataset, "LeRobotDatasetMetadata", Metadata)
    monkeypatch.setattr(
        data_loader.lerobot_dataset,
        "LeRobotDataset",
        lambda *args, **kwargs: sentinel,
    )
    return sentinel


def test_create_torch_dataset_validates_doosan_provenance_before_opening_dataset(tmp_path, monkeypatch):
    _write_provenance(tmp_path)
    sentinel = _patch_lerobot(monkeypatch, tmp_path)
    data_config = _config.DataConfig(
        repo_id="local-doosan",
        doosan_orientation_representation="rotation6d",
        doosan_state_mode="full",
        doosan_require_explicit_model_state_profile=True,
    )

    dataset = data_loader.create_torch_dataset(
        data_config,
        action_horizon=50,
        model_config=object(),
    )

    assert dataset is sentinel


def test_create_torch_dataset_fails_closed_for_profile_mismatch(tmp_path, monkeypatch):
    _write_provenance(tmp_path, representation="rotvec_continuous", state_dim=25)
    _patch_lerobot(monkeypatch, tmp_path)
    data_config = _config.DataConfig(
        repo_id="local-doosan",
        doosan_orientation_representation="rotvec_principal",
        doosan_state_mode="full",
        doosan_require_explicit_model_state_profile=True,
    )

    with pytest.raises(ValueError, match="metadata mismatch"):
        data_loader.create_torch_dataset(
            data_config,
            action_horizon=50,
            model_config=object(),
        )


def test_create_torch_dataset_rejects_incomplete_doosan_contract(tmp_path, monkeypatch):
    _patch_lerobot(monkeypatch, tmp_path)
    data_config = _config.DataConfig(
        repo_id="local-doosan",
        doosan_orientation_representation="rotation6d",
    )

    with pytest.raises(ValueError, match="configuration is incomplete"):
        data_loader.create_torch_dataset(
            data_config,
            action_horizon=50,
            model_config=object(),
        )
