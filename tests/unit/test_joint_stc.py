from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from tests.helpers.joint_stc import build_joint_fixture
from urban_intervention.representation import encoder as encoder_module
from urban_intervention.representation.dataset import RepresentationDataset, collate_samples
from urban_intervention.representation.encoder import DirectContextEncoder
from urban_intervention.representation.evaluation import (
    cosine_similarity,
    permutation_test,
    retrieval_metrics,
)
from urban_intervention.representation.export import export_embeddings
from urban_intervention.representation.joint_loss import joint_causal_loss
from urban_intervention.representation.model import JointSTCResponseModel
from urban_intervention.representation.trainer import train_representation
from urban_intervention.representation.training.joint import (
    run_joint_epoch,
    task_gradient_diagnostics,
)

OFFSETS = {"viirs": (0, 6, []), "poi": (6, 18, [])}


def batch_data():
    torch.manual_seed(17)
    return dict(
        state_tabular=torch.randn(4, 15),
        state_field_mask=torch.ones(4, 15).bool(),
        intervention=torch.randn(4, 7),
        intervention_mask=torch.ones(4, 7).bool(),
        context=torch.randn(4, 12),
        context_field_mask=torch.ones(4, 12).bool(),
        alphaearth=torch.randn(4, 64),
        alphaearth_available=torch.tensor([True, False, True, True]),
    )


def test_dinov2_prefers_cached_local_source(monkeypatch, tmp_path):
    local_root = tmp_path / "facebookresearch_dinov2_main"
    local_root.mkdir()
    calls = []
    model = torch.nn.Linear(1, 1)
    monkeypatch.setenv("TORCH_HUB", str(tmp_path))
    monkeypatch.setattr(encoder_module, "_DINO_CACHE", {})

    def fake_load(repo_or_dir, model_name, **kwargs):
        calls.append((repo_or_dir, model_name, kwargs))
        return model

    monkeypatch.setattr(torch.hub, "load", fake_load)
    assert encoder_module._ensure_dinov2() is model
    assert calls == [
        (
            str(local_root),
            "dinov2_vits14",
            {"source": "local", "skip_validation": True},
        )
    ]
    assert all(not parameter.requires_grad for parameter in model.parameters())


def losses_for(model, mode="joint", mask=None):
    z, predictions, _ = model.forward_batch(batch_data())
    targets = torch.arange(18).float()[None, :].expand(4, -1) * 0.01
    targets = targets + torch.arange(4).float()[:, None] * 0.002
    return joint_causal_loss(
        z,
        predictions,
        targets,
        torch.ones_like(targets).bool() if mask is None else mask,
        OFFSETS,
        training_mode=mode,
        response_weight=torch.ones_like(targets),
    )


def test_direct_background_shapes_identity_availability_and_scale():
    encoder = DirectContextEncoder(dropout=0).eval()
    ae = torch.randn(4, 64) * 10
    tab = torch.randn(4, 12)
    available = torch.tensor([True, False, True, False])
    b = encoder(tab, alphaearth=ae, alphaearth_available=available)
    assert b.shape == (4, 96)
    assert torch.equal(b[available, :64], ae[available])
    assert torch.equal(b[~available, :64], encoder.missing_ae_token.expand(2, -1))
    assert not torch.allclose(b.norm(dim=1), torch.ones(4))
    changed = encoder(tab, alphaearth=ae, alphaearth_available=~available)
    assert not torch.allclose(b[:, 64:], changed[:, 64:])
    ae[~available] = float("nan")
    assert torch.isfinite(encoder(tab, alphaearth=ae, alphaearth_available=available)).all()
    b.sum().backward()
    assert encoder.missing_ae_token.grad is not None
    assert encoder.missing_ae_token.grad.norm() > 0


def test_shared_shapes_and_fixed_state_invariance():
    model = JointSTCResponseModel(use_streetview=False, use_satellite=False, dropout=0).eval()
    batch = batch_data()
    z, pred, x = model.forward_batch(batch)
    assert z.shape == (4, 128) and x.shape == (4, 256)
    assert pred["viirs"].shape == (4, 6) and pred["poi"].shape == (4, 12)
    assert torch.allclose(z.norm(dim=1), torch.ones(4), atol=1e-6)
    batch["context"] += 5
    batch["intervention"] -= 3
    other_z, other_pred, _ = model.forward_batch(batch)
    assert torch.equal(z, other_z)
    assert not torch.equal(pred["viirs"], other_pred["viirs"])


@pytest.mark.parametrize("which", ["both", "viirs", "poi", "none"])
def test_active_task_anchor_normalization(which):
    z = torch.nn.functional.normalize(torch.randn(4, 128, requires_grad=True), dim=1)
    pred = {
        "viirs": torch.ones(4, 6, requires_grad=True),
        "poi": torch.full((4, 12), 2.0, requires_grad=True),
    }
    mask = torch.ones(4, 18).bool()
    if which in {"poi", "none"}:
        mask[:, :6] = False
    if which in {"viirs", "none"}:
        mask[:, 6:] = False
    result = joint_causal_loss(z, pred, torch.zeros(4, 18), mask, OFFSETS)
    expected = {"both": 1.0, "viirs": 0.5, "poi": 1.5, "none": 0.0}[which]
    assert result["anchor"].item() == pytest.approx(expected)
    assert result["active_align_tasks"] == 0  # Constant paths have no comparable pairs.
    assert result["update_active"] == (which != "none")
    result["total"].backward()


def test_alignment_has_independent_task_validity():
    model = JointSTCResponseModel(use_streetview=False, use_satellite=False, dropout=0)
    z, pred, _ = model.forward_batch(batch_data())
    target = torch.arange(18).float().repeat(4, 1)
    target[:, 6:] = 0
    result = joint_causal_loss(z, pred, target, torch.ones_like(target).bool(), OFFSETS)
    assert result["active_anchor_tasks"] == 2
    assert result["active_align_tasks"] == 1
    assert torch.equal(result["align"], result["tasks"]["viirs"]["align"])


def test_nonfinite_labels_weights_se_and_zero_pairs_are_explicit():
    z = torch.nn.functional.normalize(torch.randn(1, 128, requires_grad=True), dim=1)
    target = torch.arange(18).float()[None, :]
    target[0, :3] = float("nan")
    weights = torch.ones_like(target)
    weights[0, 3:6] = torch.tensor([0, float("nan"), float("inf")])
    pred = {
        "viirs": torch.zeros(1, 6, requires_grad=True),
        "poi": torch.zeros(1, 12, requires_grad=True),
    }
    result = joint_causal_loss(
        z,
        pred,
        target,
        torch.ones_like(target).bool(),
        OFFSETS,
        response_weight=weights,
        response_se=torch.full_like(target, float("nan")),
    )
    assert result["active_anchor_tasks"] == 1 and result["active_align_tasks"] == 0
    assert torch.isfinite(result["total"])
    result["total"].backward()
    pred["poi"] = torch.full((1, 12), float("nan"))
    with pytest.raises(FloatingPointError, match="prediction"):
        joint_causal_loss(z, pred, target, torch.ones_like(target).bool(), OFFSETS)


def test_gradient_diagnostics_do_not_change_real_update_and_detect_conflict():
    model = JointSTCResponseModel(use_streetview=False, use_satellite=False, dropout=0)
    other = copy.deepcopy(model)
    first, second = losses_for(model), losses_for(other)
    diagnostics = task_gradient_diagnostics(model, first)
    assert all(p.grad is None for p in model.parameters())
    assert diagnostics["state_grad_norm_viirs"] > 0 and diagnostics["state_grad_norm_poi"] > 0
    first["total"].backward()
    second["total"].backward()
    for a, b in zip(model.parameters(), other.parameters(), strict=True):
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad)
    for network in (model, other):
        torch.optim.SGD(network.parameters(), lr=0.01).step()
    assert all(
        torch.equal(a, b) for a, b in zip(model.parameters(), other.parameters(), strict=True)
    )
    model.zero_grad(set_to_none=True)
    z, _, _ = model.forward_batch(batch_data())
    objective = z[:, 0].sum()
    conflict = task_gradient_diagnostics(
        model,
        {
            "tasks": {
                "viirs": {"diagnostic_loss": objective},
                "poi": {"diagnostic_loss": -objective},
            }
        },
    )
    assert conflict["task_gradient_cosine"] == pytest.approx(-1, abs=1e-6)
    assert conflict["task_gradient_conflict"] is True


@pytest.mark.parametrize("mode", ["viirs_only", "poi_only"])
def test_both_tasks_can_update_same_state_encoder(mode):
    model = JointSTCResponseModel(use_streetview=False, use_satellite=False, dropout=0)
    before = {k: v.clone() for k, v in model.state_encoder.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters())
    losses_for(model, mode)["total"].backward()
    optimizer.step()
    assert any(not torch.equal(v, model.state_encoder.state_dict()[k]) for k, v in before.items())
    inactive = model.poi_head if mode == "viirs_only" else model.viirs_head
    assert all(p.grad is None for p in inactive.parameters())


def test_global_clipping_and_no_supervision_skips_optimizer(tmp_path):
    root = build_joint_fixture(tmp_path, n_units=16)
    ds = RepresentationDataset(root, split="train", load_images=False, load_satellite_images=False)
    batch = collate_samples([ds[i] for i in range(4)])
    loader = SimpleNamespace(dataset=ds)

    class Loader:
        dataset = loader.dataset

        def __iter__(self):
            yield batch

    model = JointSTCResponseModel(use_streetview=False, use_satellite=False, dropout=0)
    options = dict(
        training_mode="joint",
        anchor_weight=0.85,
        align_weight=0.15,
        delta=1.0,
        temperature=0.07,
        se_shrinkage=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)
    _, _, logs = run_joint_epoch(
        model, Loader(), torch.device("cpu"), options, optimizer=optimizer, max_grad_norm=1e-5
    )
    assert logs[0]["joint_grad_norm_before_clip"] > 1e-5
    assert logs[0]["joint_grad_norm_after_clip"] <= 1.00001e-5
    batch["response_mask"][:] = False
    before = {k: v.clone() for k, v in model.state_dict().items()}
    metrics, _, _ = run_joint_epoch(
        model, Loader(), torch.device("cpu"), options, optimizer=optimizer
    )
    assert metrics["steps"] == 0 and metrics["skipped_batches"] == 1
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())


def test_three_controlled_modes_checkpoint_export_and_fixed_support(tmp_path):
    root = build_joint_fixture(tmp_path / "data", n_units=16)
    signatures = []
    for mode in ("joint", "viirs_only", "poi_only"):
        out = tmp_path / mode
        checkpoint = train_representation(
            root,
            out,
            training_mode=mode,
            epochs=1,
            batch_size=4,
            device="cpu",
            dropout=0,
            eval_n_perm=2,
            eval_n_boot=2,
            run_baselines=True,
            run_transfer=True,
            gradient_diagnostics_interval=1,
        )
        ckpt = torch.load(checkpoint, weights_only=True)
        assert ckpt["stc_contract"]["dimensions"]["predictor_input"] == 256
        report = json.loads((out / "evaluation_report.json").read_text())
        signatures.append([report["test"]["tasks"][f]["support_sha256"] for f in ("viirs", "poi")])
        exported = export_embeddings(checkpoint, root, out / "embeddings.parquet", device="cpu")
        frame = pd.read_parquet(exported)
        assert len([c for c in frame if c.startswith("emb_")]) == 128
        assert len([c for c in frame if c.startswith("tau_hat_viirs_")]) == 6
        assert len([c for c in frame if c.startswith("tau_hat_poi_")]) == 12
        assert "station_event_id" in frame and "poi_training_mask" in frame
        model = JointSTCResponseModel(use_streetview=False, use_satellite=False, dropout=0).eval()
        model.load_state_dict(ckpt["model_state_dict"])
        ds = RepresentationDataset(root, load_images=False, load_satellite_images=False)
        batch = collate_samples([ds[0]])
        with torch.no_grad():
            z, pred, _ = model.forward_batch(batch)
        row = frame.loc[frame.treatment_order.eq(ds[0].treatment_order)].iloc[0]
        assert torch.allclose(
            z[0],
            torch.tensor(
                row[[f"emb_{i}" for i in range(128)]].to_numpy(dtype=float), dtype=torch.float32
            ),
            atol=1e-6,
        )
        assert pred["poi"][0, 0] == pytest.approx(row.tau_hat_poi_0, abs=1e-6)
    assert signatures[0] == signatures[1] == signatures[2]


def test_joint_permutation_observed_matches_retrieval():
    torch.manual_seed(42)
    y = torch.randn(16, 18)
    y[:, 6:] += torch.arange(16)[:, None] * 5
    z = torch.nn.functional.normalize(torch.randn(16, 128), dim=1)
    mask = torch.ones_like(y).bool()
    retrieval = retrieval_metrics(z, y, mask, k=2, response_offsets=OFFSETS)
    perm = permutation_test(cosine_similarity(z), y, mask, k=2, n_perm=2, response_offsets=OFFSETS)
    assert perm["observed"] == retrieval["overall"]["nn_corr@k"]


def test_full_epoch_diagnostic_switch_preserves_adam_update(tmp_path):
    root = build_joint_fixture(tmp_path, n_units=16)
    ds = RepresentationDataset(root, split="train", load_images=False, load_satellite_images=False)
    batch = collate_samples([ds[i] for i in range(4)])

    class Loader:
        dataset = ds

        def __iter__(self):
            yield copy.deepcopy(batch)

    first = JointSTCResponseModel(use_streetview=False, use_satellite=False, dropout=0)
    second = copy.deepcopy(first)
    options = dict(
        training_mode="joint",
        anchor_weight=0.85,
        align_weight=0.15,
        delta=1.0,
        temperature=0.07,
        se_shrinkage=True,
    )
    for model, interval in ((first, 1), (second, 0)):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        run_joint_epoch(
            model,
            Loader(),
            torch.device("cpu"),
            options,
            optimizer=optimizer,
            gradient_diagnostics_interval=interval,
        )
    assert all(
        torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters(), strict=True)
    )


def test_downstream_analysis_uses_only_shared_z(tmp_path):
    from urban_intervention.representation.embedding_analysis import analyze_embeddings

    torch.manual_seed(42)
    z = torch.nn.functional.normalize(torch.randn(8, 128), dim=1).numpy()
    frame = pd.DataFrame(z, columns=[f"emb_{i}" for i in range(128)])
    frame["treatment_order"] = range(8)
    frame["city_key"] = "city_a"
    frame["fused_0"] = float("nan")
    path = tmp_path / "embeddings.parquet"
    frame.to_parquet(path)
    result = analyze_embeddings(path, tmp_path / "analysis.json")
    assert result["dimensions"] == 128
    assert len(result["clusters"]) == 8
    assert len(result["manifold"]["coords_2d"]) == 8
