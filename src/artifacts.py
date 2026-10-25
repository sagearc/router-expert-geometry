"""Schemas shared by experiment writers and their analysis/plot consumers."""

from typing import Literal, NotRequired, TypedDict

from src.types import FloatArray, IntArray


class RetainedInputs(TypedDict):
    router_inputs: FloatArray
    sample_title_index: NotRequired[IntArray]
    sample_token_index: NotRequired[IntArray]


class LayerCorrelationGroup(TypedDict):
    mean: list[float]
    title_std: list[float]
    valid_titles: list[int]
    undefined_titles: list[int]


type CorrelationGroup = Literal['routed_top8', 'bottom8', 'bottom_half']


class LayerCorrelations(TypedDict):
    layers: list[int]
    n_titles: int
    response: Literal['l2_norm_w_gate_h']
    aggregation: Literal['mean_of_per_title_spearman']
    groups: dict[CorrelationGroup, LayerCorrelationGroup]


class DynamicsPoint(TypedDict):
    step: int
    routed: float
    routed_ci: list[float]
    nonrouted: float
    nonrouted_ci: list[float]
    identity_permuted_routed: float


class DynamicsFinal(TypedDict):
    routed: float
    nonrouted: float
    identity_permuted_routed: float
    token_shuffled_routed: float
    identity_permuted_routed_range: NotRequired[list[float]]
    routed_by_layer: NotRequired[list[float]]
    nonrouted_by_layer: NotRequired[list[float]]


class DynamicsResult(TypedDict):
    metric: Literal['w1_top10_l2']
    dynamics: list[DynamicsPoint]
    final: DynamicsFinal
    protocol: NotRequired[dict[str, int]]
    capture_sha256: NotRequired[dict[str, str]]


class PairedSummary(TypedDict):
    mean: float
    ci95: list[float]


class CausalCondition(TypedDict):
    sequence_ce: list[float]
    target_routes_unchanged: bool


class CausalIntervention(TypedDict):
    conditions: dict[str, CausalCondition]
    cached_full_max_error: float


class CausalLayer(TypedDict):
    interventions: dict[str, CausalIntervention]
    candidate_energies: list[list[float]]
    cached_baseline_max_error: float


class CausalProtocol(TypedDict):
    candidate_count: int
    power_iterations: int
    candidate_seed: int
    bootstrap_seed: int
    bootstrap_draws: int
    predictions_per_sequence: int
    calibration_sequences_skipped_per_domain: int
    evaluation_sequences_per_domain: int
    interventions: list[str]


class PublicCausalResult(TypedDict):
    revision: str
    transformers: str
    torch: str
    backend: str
    protocol: CausalProtocol
    sources_sha256: dict[str, str]
    sample_ids: list[str]
    sequence_sha256: list[str]
    baseline_sequence_ce: list[float]
    layers: dict[str, CausalLayer]
    complete: bool
    model_settings: dict[str, str | float]


class CausalSummaryLayer(TypedDict):
    layer_zero_based: int
    intervention: str
    router_delta_ce: float
    router_rank_descending: int
    controls_tied_with_router: int
    strongest_control: str
    router_minus_strongest_control_ce: float
    paired_ci95: list[float]
    relative_ppl_change: float


class CausalSummary(TypedDict):
    revision: str
    baseline_ce: float
    baseline_ppl: float
    layers: list[CausalSummaryLayer]
    interval_scope: str
    source_sha256: NotRequired[str]


class GeometryLayer(TypedDict):
    layer: int
    checkpoint_key: str
    experts: int
    pairs: int
    mean: float
    std: float
    min: float
    max: float


class GeometryResult(TypedDict):
    checkpoint: str
    statistic: str
    spread: str
    layers: list[GeometryLayer]


class RankMetadata(TypedDict, total=False):
    source_format: str
    model: str
    revision: str
    top_k: int
    token_scope: str
    excluded_hash_moe_layers: int
    response_metrics: list[str]
    rank_definition: str
    n_tokens: int
    vllm: str
    torch: str
    transformers: str
    prefix_caching: bool
    tokens_sha256: str
    titles_sha256: str


class RankCurve(TypedDict):
    model: str
    expert_rank: list[int]
    top_k: int
    metric_names: list[str]
    response_mean: list[list[float]]
    aggregation: str
    source_sha256: str
    capture_metadata: RankMetadata


class SpecialistCondition(TypedDict):
    experts: list[list[int]]
    sequence_nll: dict[str, list[float]]
    damage_by_domain: dict[str, float]
    damage_on: NotRequired[float]
    damage_off: NotRequired[float]
    specialization_index: NotRequired[float]


class SpecialistResult(TypedDict):
    checkpoint: str
    routing: str
    baseline: dict[str, list[float]]
    conditions: dict[str, SpecialistCondition]
    top_r: int
    random_seed: int
    sequences_sha256: str
    counts_sha256: str
