import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from scripts import run_fno_mergeable_sketch_diagnostic as diag


def _spec() -> diag.ExactStateSpec:
    return diag.ExactStateSpec(
        target_kind="toy",
        state_dim=1,
        merge_kind="sum",
        readout_kind="identity",
        state_scale=1.0,
        scalar_scale=10.0,
        precision=1,
        universe_size=10,
        cms_num_hashes=1,
        cms_num_buckets=1,
    )


def _sample() -> diag.ExactStateSample:
    leaf_a = torch.tensor([1.0]).numpy()
    leaf_b = torch.tensor([1.0]).numpy()
    parent = torch.tensor([2.0]).numpy()
    return diag.ExactStateSample(
        leaf_states=torch.stack((torch.tensor([1.0]), torch.tensor([1.0]))).numpy(),
        node_states=[leaf_a, leaf_b, parent],
        merge_pairs=[(leaf_a, leaf_b, parent)],
        root_state=parent,
        root_scalar=2.0,
        node_depths=[0, 0, 1],
        node_spans=[(0, 1), (1, 2), (0, 2)],
        node_masses=[0.5, 0.5, 1.0],
    )


def _state_transform() -> diag.StateTransform:
    return diag.StateTransform(kind="register_div64", scale=1.0)


def _scalar_transform() -> diag.ScalarTransform:
    return diag.ScalarTransform(kind="linear01", scale=10.0)


def _hll_cache_args(tmp_path) -> SimpleNamespace:
    return SimpleNamespace(
        precision=4,
        n_leaves=2,
        n_train=3,
        n_val=1,
        min_tokens=8,
        max_tokens=8,
        universe_size=32,
        zipf_alphas="1.0",
        seed=0,
        focus_token=0,
        cms_num_hashes=1,
        cms_num_buckets=8,
        sample_cache_dir=tmp_path / "cache",
    )


def _readout(state) -> float:
    return float(state[0])


class TinyRolloutModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.readout_weight = nn.Parameter(torch.tensor(0.1))
        self.merge_delta = nn.Parameter(torch.tensor(1.0))
        self.seen_predict_inputs: list[torch.Tensor] = []

    def freeze_for_f(self) -> None:
        self.readout_weight.requires_grad = True
        self.merge_delta.requires_grad = False

    def freeze_for_g(self) -> None:
        self.readout_weight.requires_grad = False
        self.merge_delta.requires_grad = True

    def merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left + right + self.merge_delta

    def predict_transformed(self, state: torch.Tensor) -> torch.Tensor:
        self.seen_predict_inputs.append(state.detach().cpu().clone())
        return state.reshape(-1) * self.readout_weight


class FiberCollisionHLLModel(nn.Module):
    def merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        exact = torch.maximum(left, right)
        bad = exact.clone()
        mask = (exact[:, 0] == 1.0) & (exact[:, 1] == 0.0)
        bad[mask, 0] = 0.0
        bad[mask, 1] = 1.0
        return bad

    def predict_transformed(self, state: torch.Tensor) -> torch.Tensor:
        return state.sum(dim=1) / 10.0


def _hll_two_register_collision_sample() -> diag.ExactStateSample:
    leaf_a = torch.tensor([1.0, 0.0]).numpy()
    leaf_b = torch.tensor([0.0, 0.0]).numpy()
    leaf_c = torch.tensor([0.0, 1.0]).numpy()
    leaf_d = torch.tensor([0.0, 0.0]).numpy()
    parent_left = torch.tensor([1.0, 0.0]).numpy()
    parent_right = torch.tensor([0.0, 1.0]).numpy()
    root = torch.tensor([1.0, 1.0]).numpy()
    return diag.ExactStateSample(
        leaf_states=torch.stack(
            (
                torch.tensor([1.0, 0.0]),
                torch.tensor([0.0, 0.0]),
                torch.tensor([0.0, 1.0]),
                torch.tensor([0.0, 0.0]),
            )
        ).numpy(),
        node_states=[leaf_a, leaf_b, leaf_c, leaf_d, parent_left, parent_right, root],
        node_scalars=[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0],
        merge_pairs=[
            (leaf_a, leaf_b, parent_left),
            (leaf_c, leaf_d, parent_right),
            (parent_left, parent_right, root),
        ],
        root_state=root,
        root_scalar=2.0,
        node_depths=[0, 0, 0, 0, 1, 1, 2],
        node_spans=[(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (2, 4), (0, 4)],
        node_masses=[0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 1.0],
    )


def test_rollout_preserves_topology_and_uses_learned_parent_state() -> None:
    model = TinyRolloutModel()
    batch = diag._rollout_batch(
        model,
        [_sample()],
        spec=_spec(),
        device=torch.device("cpu"),
        state_transform=_state_transform(),
        scalar_transform=_scalar_transform(),
        readout=_readout,
        detach_merge_states=True,
    )

    assert tuple(batch.states.reshape(-1).tolist()) == pytest.approx((1.0, 1.0, 3.0))
    assert tuple(batch.exact_states.reshape(-1).tolist()) == pytest.approx((1.0, 1.0, 2.0))
    assert tuple(batch.targets.tolist()) == pytest.approx((0.1, 0.1, 0.2))
    assert batch.root_indices.tolist() == [2]
    assert batch.merge_indices.tolist() == [2]
    assert batch.observed.tolist() == [False, False, True]
    assert batch.propensity.tolist() == pytest.approx([0.0, 0.0, 1.0])


def test_evaluate_reports_leaf_merge_and_root_diagnostics() -> None:
    model = TinyRolloutModel()
    metrics = diag._evaluate(
        model,
        [_sample()],
        spec=_spec(),
        device=torch.device("cpu"),
        state_transform=_state_transform(),
        scalar_transform=_scalar_transform(),
        readout=_readout,
    )

    assert metrics["leaf_readout_mae"] == pytest.approx(0.0)
    assert metrics["root_readout_mae"] == pytest.approx(0.0)
    assert metrics["all_node_readout_mae"] == pytest.approx(0.0)
    assert metrics["merge_state_root_mae"] == pytest.approx(1.0)
    assert metrics["merge_readout_root_mae"] == pytest.approx(1.0)
    assert metrics["official_merge_readout_root_mae"] == pytest.approx(1.0)
    assert metrics["merge_state_mae_depth_1"] == pytest.approx(1.0)
    assert metrics["merge_readout_mae_depth_1"] == pytest.approx(1.0)
    assert metrics["learned_state_min"] == pytest.approx(1.0)
    assert metrics["learned_state_median"] == pytest.approx(1.0)
    assert metrics["learned_state_max"] == pytest.approx(3.0)
    assert metrics["learned_state_above_valid_frac"] == pytest.approx(1.0 / 3.0)
    assert metrics["learned_root_state_above_valid_frac"] == pytest.approx(1.0)
    assert metrics["hll_readout_postclamp_near_one_frac"] == pytest.approx(0.0)


def test_evaluate_reports_hll_scalar_correct_state_wrong_alignment() -> None:
    metrics = diag._evaluate(
        FiberCollisionHLLModel(),
        [_hll_two_register_collision_sample()],
        spec=diag.ExactStateSpec(
            target_kind="hll_register_space",
            state_dim=2,
            merge_kind="max_union",
            readout_kind="hll_reference",
            state_scale=1.0,
            scalar_scale=10.0,
            precision=1,
            universe_size=10,
            cms_num_hashes=1,
            cms_num_buckets=1,
        ),
        device=torch.device("cpu"),
        state_transform=diag.StateTransform(kind="register_div64", scale=1.0),
        scalar_transform=_scalar_transform(),
        readout=lambda state: float(state[0] + state[1]),
    )

    assert metrics["hll_register_fractional_abs_mean"] == pytest.approx(0.0)
    assert metrics["hll_merge_register_exact_frac"] == pytest.approx(1.0 / 3.0)
    assert metrics["hll_zero_scalar_bad_state_frac"] == pytest.approx(1.0 / 3.0)
    assert metrics["hll_future_context_readout_mae"] == pytest.approx(0.5)
    assert metrics["hll_zero_scalar_bad_future_frac"] == pytest.approx(0.5)


def test_state_validity_metrics_detect_invalid_values() -> None:
    metrics = diag._state_validity_metrics(
        torch.tensor([-1.0, 0.25, 2.0, float("nan")]).numpy(),
        prefix="state",
        valid_min=0.0,
        valid_max=1.0,
    )

    assert metrics["state_min"] == pytest.approx(-1.0)
    assert metrics["state_median"] == pytest.approx(0.25)
    assert metrics["state_max"] == pytest.approx(2.0)
    assert metrics["state_below_valid_frac"] == pytest.approx(0.25)
    assert metrics["state_above_valid_frac"] == pytest.approx(0.25)
    assert metrics["state_nonfinite_frac"] == pytest.approx(0.25)


def test_merge_output_constraint_unit_clamps_learned_states() -> None:
    values = torch.tensor([[-1.0, 0.25, 2.0]])

    clamped = diag._apply_merge_output_constraint(values, "unit_clamp")

    assert clamped.reshape(-1).tolist() == pytest.approx([0.0, 0.25, 1.0])
    assert diag._apply_merge_output_constraint(values, "none") is values


def test_resolved_model_widths_are_at_least_2x_state_input() -> None:
    args = SimpleNamespace(
        hidden_channels=128,
        head_hidden_dim=128,
        n_modes=512,
    )
    spec = diag.ExactStateSpec(
        target_kind="hll_register_space",
        state_dim=256,
        merge_kind="max_union",
        readout_kind="hll_reference",
        state_scale=64.0,
        scalar_scale=1024.0,
        precision=8,
        universe_size=512,
        cms_num_hashes=4,
        cms_num_buckets=128,
    )

    widths = diag._resolve_model_widths(args, spec)

    assert widths["hidden_width_floor"] == 512
    assert widths["head_width_floor"] == 512
    assert widths["hidden_channels"] >= 2 * spec.state_dim
    assert widths["head_hidden_dim"] >= 2 * spec.state_dim
    assert widths["width_floor_multiplier"] == 2


def test_sketch_state_fno_uses_induced_additive_projection() -> None:
    model = diag.SketchStateFNO(
        state_dim=2,
        hidden_channels=4,
        n_modes=1,
        n_layers=1,
        head_hidden_dim=4,
        readout_arch="hll_formula",
        bounded_output=True,
        state_value_scale=1.0,
        target_transform_kind="linear01",
        target_scale=10.0,
        target_mean=0.0,
        target_std=1.0,
    )
    model.initialize_residuals_as_identity()

    left = torch.tensor([[0.1, 0.8]])
    right = torch.tensor([[0.4, 0.3]])
    carrier, residual, merged = model.merge_components(left, right)

    assert model.merge_adapter == "induced_projection"
    assert carrier.reshape(-1).tolist() == pytest.approx([0.5, 1.1])
    assert residual.reshape(-1).tolist() == pytest.approx([0.0, 0.0])
    assert merged.reshape(-1).tolist() == pytest.approx([0.5, 1.1])


def test_report_writer_preserves_diagnostic_summary_fields(tmp_path) -> None:
    row = {
        "target_kind": "hll_register_space",
        "state_dim": 4,
        "schedule": "fgfg",
        "objective_mode": "rollout_local_law",
        "objective_ablation": False,
        "root_mae": 1.0,
        "root_rel_mae": 0.1,
        "merge_state_mae": 0.2,
        "wall_seconds": 3.0,
        "learned_state_above_valid_frac": 0.25,
        "hll_readout_postclamp_near_one_frac": 1.0,
        "train_local_ipw_correction_leaf_end": 0.5,
    }

    diag._write_report([row], tmp_path)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary[0]["learned_state_above_valid_frac"] == pytest.approx(0.25)
    assert summary[0]["hll_readout_postclamp_near_one_frac"] == pytest.approx(1.0)
    assert summary[0]["train_local_ipw_correction_leaf_end"] == pytest.approx(0.5)


def test_exact_state_samples_are_cached_with_node_oracle_scores(tmp_path) -> None:
    args = _hll_cache_args(tmp_path)
    spec = diag._target_spec(args, "hll_register_space")

    train, val = diag._generate_samples(args, spec)

    assert len(train) == 3
    assert len(val) == 1
    assert args._sample_cache_last["sample_cache_status"] == "miss"
    cache_path = args._sample_cache_last["sample_cache_path"]
    assert cache_path
    assert len(list((tmp_path / "cache").glob("*.pkl"))) == 1
    assert train[0].node_scalars is not None
    assert len(train[0].node_scalars) == len(train[0].node_states)

    args2 = _hll_cache_args(tmp_path)
    train2, val2 = diag._generate_samples(args2, spec)

    assert args2._sample_cache_last["sample_cache_status"] == "hit"
    assert args2._sample_cache_last["sample_cache_path"] == cache_path
    assert train2[0].root_scalar == pytest.approx(train[0].root_scalar)
    assert val2[0].node_scalars == pytest.approx(val[0].node_scalars)


def test_rollout_observation_modes_encode_sparse_and_dense_oracles() -> None:
    model = TinyRolloutModel()
    root_only = diag._rollout_batch(
        model,
        [_sample()],
        spec=_spec(),
        device=torch.device("cpu"),
        state_transform=_state_transform(),
        scalar_transform=_scalar_transform(),
        readout=_readout,
        detach_merge_states=True,
        oracle_observation_design="root_only",
    )
    sampled_all = diag._rollout_batch(
        model,
        [_sample()],
        spec=_spec(),
        device=torch.device("cpu"),
        state_transform=_state_transform(),
        scalar_transform=_scalar_transform(),
        readout=_readout,
        detach_merge_states=True,
        oracle_observation_design="sampled_nodes",
        sampled_node_rate=1.0,
    )
    dense = diag._rollout_batch(
        model,
        [_sample()],
        spec=_spec(),
        device=torch.device("cpu"),
        state_transform=_state_transform(),
        scalar_transform=_scalar_transform(),
        readout=_readout,
        detach_merge_states=True,
        oracle_observation_design="dense_oracle",
    )

    assert root_only.observed.tolist() == [False, False, True]
    assert sampled_all.observed.tolist() == [True, True, True]
    assert dense.observed.tolist() == [True, True, True]
    assert sampled_all.propensity.tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_sampled_nodes_root_and_nonroot_propensities_are_logged() -> None:
    sample = _sample()
    observed, propensity = diag._node_observation_design(
        sample,
        mode="sampled_nodes",
        sampled_node_rate=0.25,
        sampled_node_seed=11,
    )

    assert observed[-1] is True
    assert propensity[-1] == pytest.approx(1.0)
    assert len(observed) == len(sample.node_states)
    for is_observed, pi in zip(observed[:-1], propensity[:-1]):
        if is_observed:
            assert pi == pytest.approx(0.25)
        else:
            assert pi == pytest.approx(0.0)

    observed_all, propensity_all = diag._node_observation_design(
        sample,
        mode="sampled_nodes",
        sampled_node_rate=1.0,
        sampled_node_seed=11,
    )
    assert observed_all == [True, True, True]
    assert propensity_all == pytest.approx([1.0, 1.0, 1.0])


def test_sampled_observation_designs_use_common_random_numbers_across_rates() -> None:
    sample = _sample()
    low_nodes, _ = diag._node_observation_design(
        sample,
        mode="sampled_nodes",
        sampled_node_rate=0.25,
        sampled_node_seed=123,
    )
    high_nodes, _ = diag._node_observation_design(
        sample,
        mode="sampled_nodes",
        sampled_node_rate=0.75,
        sampled_node_seed=123,
    )
    assert all((not low) or high for low, high in zip(low_nodes, high_nodes))

    low_root_nodes, _ = diag._node_observation_design(
        sample,
        mode="sampled_root_nodes",
        sampled_node_rate=0.25,
        sampled_node_seed=123,
        root_label_share=0.10,
    )
    high_root_nodes, _ = diag._node_observation_design(
        sample,
        mode="sampled_root_nodes",
        sampled_node_rate=0.75,
        sampled_node_seed=123,
        root_label_share=0.50,
    )
    assert all((not low) or high for low, high in zip(low_root_nodes, high_root_nodes))


def test_observation_payload_omits_inactive_optional_knobs() -> None:
    root_args = SimpleNamespace(
        oracle_observation_design="root_only",
        sampled_node_rate=None,
        root_label_share=1.0,
        mass_target_per_doc=1.0,
        local_label_pool="nonroot",
        local_label_allocation="span_mass",
    )
    dense_args = SimpleNamespace(
        oracle_observation_design="dense_oracle",
        sampled_node_rate=0.5,
        root_label_share=1.0,
        mass_target_per_doc=1.0,
        local_label_pool="nonroot",
        local_label_allocation="span_mass",
    )
    fixed_args = SimpleNamespace(
        oracle_observation_design="budgeted_mass",
        sampled_node_rate=None,
        root_label_share=0.25,
        mass_target_per_doc=1.0,
        local_label_pool="leaves",
        local_label_allocation="span_mass",
    )

    assert diag._oracle_observation_payload(root_args) == {
        "oracle_observation_design": {
            "schema_version": "ctreepo.oracle_observation.v1",
            "design_id": "root_only",
            "design_parameters": {},
        }
    }
    assert diag._oracle_observation_payload(dense_args) == {
        "oracle_observation_design": {
            "schema_version": "ctreepo.oracle_observation.v1",
            "design_id": "dense_oracle",
            "design_parameters": {},
        }
    }
    assert diag._oracle_observation_payload(fixed_args) == {
        "oracle_observation_design": {
            "schema_version": "ctreepo.oracle_observation.v1",
            "design_id": "budgeted_mass",
            "design_parameters": {
                "root_label_share": 0.25,
                "mass_target_per_doc": 1.0,
                "local_label_pool": "leaves",
                "local_label_allocation": "span_mass",
            },
        }
    }


def test_sampled_node_rate_is_required_for_sampled_cli_mode(tmp_path) -> None:
    with pytest.raises(ValueError, match="--sampled-node-rate is required"):
        diag.main(
            [
                "--oracle-observation-design",
                "sampled_nodes",
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_legacy_observation_mode_cli_flag_is_rejected(tmp_path) -> None:
    with pytest.raises(SystemExit):
        diag.main(
            [
                "--oracle-observation-" + "mode",
                "sampled_nodes",
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_sampled_observation_payload_emits_supplied_rate() -> None:
    args = SimpleNamespace(
        oracle_observation_design="sampled_nodes",
        sampled_node_rate=0.5,
        root_label_share=1.0,
        mass_target_per_doc=1.0,
        local_label_pool="nonroot",
        local_label_allocation="span_mass",
    )
    args._sampled_node_rate_internal = diag._resolve_sampled_node_rate(args)

    assert diag._oracle_observation_payload(args) == {
        "oracle_observation_design": {
            "schema_version": "ctreepo.oracle_observation.v1",
            "design_id": "sampled_nodes",
            "design_parameters": {"sampled_node_rate": 0.5},
        }
    }


def test_budgeted_mass_observation_design_encodes_r100_and_r0() -> None:
    model = TinyRolloutModel()
    r100 = diag._rollout_batch(
        model,
        [_sample()],
        spec=_spec(),
        device=torch.device("cpu"),
        state_transform=_state_transform(),
        scalar_transform=_scalar_transform(),
        readout=_readout,
        detach_merge_states=True,
        oracle_observation_design="budgeted_mass",
        root_label_share=1.0,
        mass_target_per_doc=1.0,
    )
    r0 = diag._rollout_batch(
        model,
        [_sample()],
        spec=_spec(),
        device=torch.device("cpu"),
        state_transform=_state_transform(),
        scalar_transform=_scalar_transform(),
        readout=_readout,
        detach_merge_states=True,
        oracle_observation_design="budgeted_mass",
        root_label_share=0.0,
        mass_target_per_doc=1.0,
    )

    assert r100.observed.tolist() == [False, False, True]
    assert r100.propensity.tolist() == pytest.approx([0.0, 0.0, 1.0])
    assert r100.node_masses.tolist() == pytest.approx([0.5, 0.5, 1.0])
    assert r0.observed.tolist() == [True, True, False]
    assert r0.propensity.tolist() == pytest.approx([1.0, 1.0, 0.0])
    assert float((r0.node_masses * r0.observed.to(dtype=torch.float32)).sum()) == pytest.approx(1.0)


def test_budgeted_mass_r50_keeps_doc_mass_constant_for_root_and_local_cases() -> None:
    root_seen_sample = None
    root_missing_sample = None
    for sample_id in range(1000):
        sample = _sample()
        sample.sample_id = sample_id
        observed, propensity = diag._node_observation_design(
            sample,
            mode="budgeted_mass",
            sampled_node_rate=0.0,
            sampled_node_seed=7,
            root_label_share=0.5,
            mass_target_per_doc=1.0,
            local_label_pool="nonroot",
            local_label_allocation="span_mass",
        )
        if observed == [False, False, True]:
            root_seen_sample = (observed, propensity, sample)
        if observed == [True, True, False]:
            root_missing_sample = (observed, propensity, sample)
        if root_seen_sample is not None and root_missing_sample is not None:
            break

    assert root_seen_sample is not None
    assert root_missing_sample is not None
    for observed, propensity, sample in (root_seen_sample, root_missing_sample):
        masses = diag._sample_node_masses(sample)
        observed_mass = sum(mass for mass, is_observed in zip(masses, observed) if is_observed)
        assert observed_mass == pytest.approx(1.0)
        for is_observed, pi in zip(observed, propensity):
            if is_observed:
                assert pi == pytest.approx(0.5)
            else:
                assert pi == pytest.approx(0.0)


def test_rollout_f_stage_trains_on_current_g_states_not_exact_rows() -> None:
    model = TinyRolloutModel()
    diag._train_f_stage_rollout(
        model,
        samples=[_sample()],
        spec=_spec(),
        device=torch.device("cpu"),
        state_transform=_state_transform(),
        scalar_transform=_scalar_transform(),
        readout=_readout,
        eval_callback=None,
        epochs=1,
        batch_size=16,
        rollout_min_docs_per_batch=1,
        rollout_max_docs_per_batch=0,
        learning_rate=0.0,
        weight_decay=0.0,
        grad_clip_norm=0.0,
        grad_accum_steps=1,
        local_law_weight=1.0,
        local_law_leaf_discount_gamma=1.0,
        objective_loss_weight=1.0,
        exact_state_anchor_weight=0.0,
    )

    seen = torch.cat(model.seen_predict_inputs, dim=0).reshape(-1).tolist()
    assert seen == pytest.approx([1.0, 1.0, 3.0])


def test_rollout_row_batches_respect_min_doc_floor() -> None:
    samples = [_sample() for _ in range(5)]

    batches = list(
        diag._sample_batches_by_node_rows(
            samples,
            batch_size=1,
            min_docs_per_batch=2,
            max_docs_per_batch=0,
        )
    )

    assert [len(batch) for batch in batches] == [2, 2, 1]


def test_g_rollout_loss_flows_to_merge_but_not_f_params() -> None:
    model = TinyRolloutModel()
    model.freeze_for_g()
    batch = diag._rollout_batch(
        model,
        [_sample()],
        spec=_spec(),
        device=torch.device("cpu"),
        state_transform=_state_transform(),
        scalar_transform=_scalar_transform(),
        readout=_readout,
        detach_merge_states=False,
    )
    predictions = model.predict_transformed(batch.states)
    loss = diag._single_lambda_rollout_loss(
        predictions,
        batch.targets,
        root_indices=batch.root_indices,
        local_law_weight=1.0,
    ).loss
    loss.backward()

    assert model.merge_delta.grad is not None
    assert float(model.merge_delta.grad.abs()) > 0.0
    assert model.readout_weight.grad is None


def test_single_lambda_rollout_loss_endpoints() -> None:
    predictions = torch.tensor([0.0, 2.0, 4.0])
    targets = torch.tensor([1.0, 1.0, 1.0])
    root_indices = torch.tensor([2])

    root_only = diag._single_lambda_rollout_loss(
        predictions,
        targets,
        root_indices=root_indices,
        local_law_weight=0.0,
    )
    local_only = diag._single_lambda_rollout_loss(
        predictions,
        targets,
        root_indices=root_indices,
        local_law_weight=1.0,
    )

    assert float(root_only.loss) == pytest.approx(9.0)
    assert float(root_only.root_loss) == pytest.approx(9.0)
    assert float(local_only.loss) == pytest.approx(11.0 / 3.0)
    assert float(local_only.local_loss) == pytest.approx(11.0 / 3.0)


def test_local_law_leaf_discount_weights_are_root_distance_weights() -> None:
    depths = torch.tensor([0, 0, 0, 0, 1, 1, 2])
    root_indices = torch.tensor([6])

    distance_depths = diag._distance_from_root_depths(
        depths,
        root_indices=root_indices,
        row_count=7,
    )
    weights = torch.pow(torch.full((7,), 0.8), distance_depths.to(dtype=torch.float32))

    assert distance_depths.tolist() == [2, 2, 2, 2, 1, 1, 0]
    assert weights.tolist() == pytest.approx([0.64, 0.64, 0.64, 0.64, 0.8, 0.8, 1.0])


def test_single_lambda_rollout_loss_changes_with_leaf_discount_gamma() -> None:
    predictions = torch.zeros(7)
    oracle_targets = torch.tensor([2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
    root_indices = torch.tensor([6])
    observed = torch.ones(7, dtype=torch.bool)
    propensity = torch.ones(7)
    depths = torch.tensor([0, 0, 0, 0, 1, 1, 2])

    undiscounted = diag._single_lambda_rollout_loss(
        predictions,
        oracle_targets,
        root_indices=root_indices,
        local_law_weight=1.0,
        proxy_targets=predictions.detach().clone(),
        oracle_targets=oracle_targets,
        observed=observed,
        propensity=propensity,
        depths=depths,
        local_law_leaf_discount_gamma=1.0,
    )
    discounted = diag._single_lambda_rollout_loss(
        predictions,
        oracle_targets,
        root_indices=root_indices,
        local_law_weight=1.0,
        proxy_targets=predictions.detach().clone(),
        oracle_targets=oracle_targets,
        observed=observed,
        propensity=propensity,
        depths=depths,
        local_law_leaf_discount_gamma=0.8,
    )

    assert float(undiscounted.local_loss) == pytest.approx(19.0 / 7.0)
    assert float(discounted.local_loss) == pytest.approx(12.84 / 5.16)
    assert float(discounted.local_loss) != pytest.approx(float(undiscounted.local_loss))
    assert discounted.discounted_root_weight == pytest.approx(1.0)
    assert discounted.discounted_nonroot_weight == pytest.approx(4.16)


def test_single_lambda_rollout_loss_uses_corrected_rows() -> None:
    predictions = torch.tensor([0.0, 2.0, 4.0])
    proxy_targets = predictions.detach().clone()
    oracle_targets = torch.tensor([1.0, 1.0, 1.0])
    root_indices = torch.tensor([2])
    observed = torch.tensor([False, False, True])
    propensity = torch.tensor([0.0, 0.0, 1.0])
    depths = torch.tensor([0, 0, 1])

    loss = diag._single_lambda_rollout_loss(
        predictions,
        oracle_targets,
        root_indices=root_indices,
        local_law_weight=1.0,
        proxy_targets=proxy_targets,
        oracle_targets=oracle_targets,
        observed=observed,
        propensity=propensity,
        depths=depths,
    )

    assert float(loss.root_loss) == pytest.approx(9.0)
    assert float(loss.local_loss) == pytest.approx(3.0)
    assert loss.local_proxy_loss == pytest.approx(0.0)
    assert loss.local_oracle_observed_ipw_loss == pytest.approx(3.0)
    assert loss.local_ipw_correction == pytest.approx(3.0)
    assert loss.local_corrected_loss == pytest.approx(3.0)
    assert loss.discounted_root_weight == pytest.approx(1.0)
    assert loss.discounted_nonroot_weight == pytest.approx(2.0)
    assert loss.observed_count == 1
    assert loss.population_count == 3
    assert loss.nonroot_observed_count == 0
    assert loss.nonroot_population_count == 2
    assert loss.observed_rows_per_doc == pytest.approx(1.0)
    assert loss.root_observed_rows_per_doc == pytest.approx(1.0)
    assert loss.nonroot_observed_rows_per_doc == pytest.approx(0.0)
    assert loss.max_ipw_weight == pytest.approx(1.0)
    assert loss.effective_sample_size == pytest.approx(1.0)


def test_single_lambda_rollout_loss_reports_sampled_nonroot_ipw_accounting() -> None:
    predictions = torch.tensor([0.0, 2.0, 4.0])
    oracle_targets = torch.tensor([1.0, 1.0, 1.0])
    root_indices = torch.tensor([2])
    observed = torch.tensor([True, False, True])
    propensity = torch.tensor([0.25, 0.0, 1.0])

    loss = diag._single_lambda_rollout_loss(
        predictions,
        oracle_targets,
        root_indices=root_indices,
        local_law_weight=1.0,
        proxy_targets=predictions.detach().clone(),
        oracle_targets=oracle_targets,
        observed=observed,
        propensity=propensity,
    )

    assert loss.observed_count == 2
    assert loss.nonroot_observed_count == 1
    assert loss.root_observed_count == 1
    assert loss.nonroot_population_count == 2
    assert loss.observed_rows_per_doc == pytest.approx(2.0)
    assert loss.nonroot_observed_rows_per_doc == pytest.approx(1.0)
    assert loss.max_ipw_weight == pytest.approx(4.0)
    assert loss.effective_sample_size == pytest.approx(25.0 / 17.0)


def test_root_loss_ignores_unobserved_root_labels() -> None:
    predictions = torch.tensor([0.0, 2.0, 4.0])
    targets = torch.tensor([1.0, 1.0, 1.0])
    root_indices = torch.tensor([2])
    observed = torch.tensor([True, True, False])
    propensity = torch.tensor([1.0, 1.0, 0.0])

    loss = diag._single_lambda_rollout_loss(
        predictions,
        targets,
        root_indices=root_indices,
        local_law_weight=0.0,
        observed=observed,
        propensity=propensity,
    )

    assert float(loss.root_loss) == pytest.approx(0.0)
    assert float(loss.loss) == pytest.approx(0.0)
    assert loss.root_observed_count == 0
    assert loss.root_population_count == 1


def test_exact_rows_stage_keeps_legacy_direct_state_training_path() -> None:
    model = TinyRolloutModel()
    states = torch.tensor([[7.0], [8.0]])
    targets = torch.tensor([0.7, 0.8])
    diag._train_f_stage(
        model,
        states=states,
        targets=targets,
        eval_callback=None,
        epochs=1,
        batch_size=16,
        learning_rate=0.0,
        weight_decay=0.0,
        grad_clip_norm=0.0,
        grad_accum_steps=1,
    )

    seen = sorted(torch.cat(model.seen_predict_inputs, dim=0).reshape(-1).tolist())
    assert seen == pytest.approx([7.0, 8.0])


def test_training_progress_line_is_flushed_and_metric_bearing(capsys) -> None:
    diag._emit_epoch_progress(
        "target=toy stage=1/1 component=g",
        epoch=1,
        epochs=2,
        train_loss=0.25,
        metrics={"root_mae": 1.5, "merge_state_mae": 0.125},
        started_at=0.0,
        progress_every_epochs=1,
    )

    out = capsys.readouterr().out
    assert "[fno-sketch] target=toy stage=1/1 component=g epoch=1/2" in out
    assert "train_loss=0.25" in out
    assert "root_mae=1.5" in out
    assert "merge_state_mae=0.125" in out


def test_batch_progress_line_is_flushed(capsys) -> None:
    diag._emit_batch_progress(
        "target=toy stage=1/1 component=g",
        epoch=1,
        epochs=2,
        batch_index=5,
        batches=9,
        loss=0.125,
        started_at=0.0,
        progress_every_batches=5,
    )

    out = capsys.readouterr().out
    assert "epoch=1/2" in out
    assert "batch=5/9" in out
    assert "loss=0.125" in out


def test_fail_fast_rejects_nonfinite_training_values() -> None:
    with pytest.raises(RuntimeError, match="non-finite loss"):
        diag._assert_finite_scalar("loss", float("nan"), context="unit")


def test_training_stage_fails_before_silent_empty_epochs() -> None:
    model = TinyRolloutModel()

    with pytest.raises(RuntimeError, match="no training batches"):
        diag._train_f_stage(
            model,
            states=torch.empty((0, 1)),
            targets=torch.empty((0,)),
            eval_callback=None,
            epochs=1,
            batch_size=16,
            learning_rate=0.0,
            weight_decay=0.0,
            grad_clip_norm=0.0,
            grad_accum_steps=1,
            progress_label="unit",
        )


def test_cli_rejects_bad_batch_size_before_building_samples(tmp_path) -> None:
    with pytest.raises(ValueError, match="--batch-size must be positive"):
        diag.main(["--batch-size", "0", "--output-dir", str(tmp_path)])
