from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

MetricKind = Literal["gauge", "counter", "histogram"]

ENGINE_METRICS: dict[str, MetricKind] = {
    "sglang:num_running_reqs": "gauge",
    "sglang:num_queue_reqs": "gauge",
    "sglang:num_grammar_queue_reqs": "gauge",
    "sglang:num_used_tokens": "gauge",
    "sglang:token_usage": "gauge",
    "sglang:full_token_usage": "gauge",
    "sglang:max_total_num_tokens": "gauge",
    "sglang:kv_used_tokens": "gauge",
    "sglang:kv_available_tokens": "gauge",
    "sglang:kv_evictable_tokens": "gauge",
    "sglang:pending_prealloc_token_usage": "gauge",
    "sglang:utilization": "gauge",
    "sglang:fwd_occupancy": "gauge",
    "sglang:http_requests_active": "gauge",
    "sglang:decode_sum_seq_lens": "gauge",
    "sglang:new_token_ratio": "gauge",
    "sglang:max_running_requests_under_SLO": "gauge",
    "sglang:prompt_tokens_total": "counter",
    "sglang:generation_tokens_total": "counter",
    "sglang:cached_tokens_total": "counter",
    "sglang:realtime_tokens_total": "counter",
    "sglang:gen_throughput": "gauge",
    "sglang:num_requests_total": "counter",
    "sglang:http_requests_total": "counter",
    "sglang:http_responses_total": "counter",
    "sglang:get_loads_duration_seconds": "histogram",
    "sglang:num_so_requests_total": "counter",
    "sglang:num_aborted_requests_total": "counter",
    "sglang:cache_hit_rate": "gauge",
    "sglang:time_to_first_token_seconds": "histogram",
    "sglang:inter_token_latency_seconds": "histogram",
    "sglang:e2e_request_latency_seconds": "histogram",
    "sglang:queue_time_seconds": "histogram",
    "sglang:per_stage_req_latency_seconds": "histogram",
    "sglang:prompt_tokens_histogram": "histogram",
    "sglang:uncached_prompt_tokens_histogram": "histogram",
    "sglang:generation_tokens_histogram": "histogram",
    "sglang:func_latency_seconds": "histogram",
    "sglang:num_retracted_reqs": "gauge",
    "sglang:num_retracted_requests_total": "counter",
    "sglang:num_retracted_input_tokens_total": "counter",
    "sglang:num_retracted_output_tokens_total": "counter",
    "sglang:num_paused_reqs": "gauge",
    "sglang:evicted_tokens_total": "counter",
    "sglang:eviction_duration_seconds": "histogram",
    "sglang:load_back_tokens_total": "counter",
    "sglang:load_back_duration_seconds": "histogram",
    "sglang:is_cuda_graph": "gauge",
    "sglang:cuda_graph_passes_total": "counter",
    "sglang:forward_execution_seconds_total": "counter",
    "sglang:process_cpu_seconds_total": "counter",
    "sglang:engine_startup_time": "gauge",
    "sglang:engine_load_weights_time": "gauge",
    "sglang:weight_load_duration_seconds": "gauge",
    "sglang:startup_latency_breakdown_seconds_max": "gauge",
    "sglang:startup_available_gpu_memory_gb": "gauge",
    "sglang:context_len": "gauge",
    "sglang:page_size": "gauge",
    "sglang:num_pages": "gauge",
    "sglang:routing_keys_active": "gauge",
    "sglang:num_unique_running_routing_keys": "gauge",
    "sglang:routing_key_running_req_count": "gauge",
    "sglang:routing_key_all_req_count": "gauge",
    "sglang:swa_token_usage": "gauge",
    "sglang:swa_used_tokens": "gauge",
    "sglang:swa_available_tokens": "gauge",
    "sglang:swa_evictable_tokens": "gauge",
    "sglang:mamba_usage": "gauge",
    "sglang:mamba_used_tokens": "gauge",
    "sglang:mamba_available_tokens": "gauge",
    "sglang:mamba_evictable_tokens": "gauge",
    "sglang:spec_accept_length": "gauge",
    "sglang:spec_accept_rate": "gauge",
    "sglang:spec_num_steps": "gauge",
    "sglang:spec_num_draft_tokens": "gauge",
    "sglang:spec_verify_calls_total": "counter",
    "sglang:grammar_compilation_time_seconds": "histogram",
    "sglang:num_grammar_cache_hit_total": "counter",
    "sglang:num_grammar_aborted_total": "counter",
    "sglang:num_grammar_timeout_total": "counter",
    "sglang:num_grammar_total": "counter",
    "sglang:grammar_schema_count": "histogram",
    "sglang:grammar_ebnf_size": "histogram",
    "sglang:grammar_tree_traversal_time_avg": "histogram",
    "sglang:grammar_tree_traversal_time_max": "histogram",
    "sglang:lora_pool_slots_used": "gauge",
    "sglang:lora_pool_slots_total": "gauge",
    "sglang:lora_pool_utilization": "gauge",
    "sglang:prefill_delayer_wait_forward_passes": "histogram",
    "sglang:prefill_delayer_wait_seconds": "histogram",
    "sglang:prefill_delayer_outcomes_total": "counter",
    "sglang:eplb_balancedness": "histogram",
    "sglang:eplb_gpu_physical_count": "histogram",
    "sglang:dp_cooperation_realtime_tokens_total": "counter",
    "sglang:dp_cooperation_forward_execution_seconds_total": "counter",
    "sglang:num_prefill_bootstrap_queue_reqs": "gauge",
    "sglang:num_prefill_inflight_queue_reqs": "gauge",
    "sglang:num_decode_prealloc_queue_reqs": "gauge",
    "sglang:num_decode_transfer_queue_reqs": "gauge",
    "sglang:num_bootstrap_failed_reqs_total": "counter",
    "sglang:num_transfer_failed_reqs_total": "counter",
    "sglang:failed_session_recoveries_total": "counter",
    "sglang:num_prefill_retries_total": "counter",
    "sglang:kv_transfer_speed_gb_s": "histogram",
    "sglang:kv_transfer_latency_ms": "histogram",
    "sglang:kv_transfer_bootstrap_ms": "histogram",
    "sglang:kv_transfer_alloc_ms": "histogram",
    "sglang:kv_transfer_total_mb": "histogram",
    "sglang:prefetched_tokens_total": "counter",
    "sglang:backuped_tokens_total": "counter",
    "sglang:prefetch_pgs": "histogram",
    "sglang:backup_pgs": "histogram",
    "sglang:prefetch_bandwidth": "histogram",
    "sglang:backup_bandwidth": "histogram",
    "sglang:hicache_host_total_tokens": "gauge",
    "sglang:hicache_host_used_tokens": "gauge",
    "sglang:num_streaming_sessions": "gauge",
    "sglang:streaming_session_held_tokens": "gauge",
    "sglang:estimated_flops_per_gpu_total": "counter",
    "sglang:estimated_read_bytes_per_gpu_total": "counter",
    "sglang:estimated_write_bytes_per_gpu_total": "counter",
}

ROUTER_METRICS: dict[str, MetricKind] = {
    "smg_http_requests_total": "counter",
    "smg_http_responses_total": "counter",
    "smg_http_request_duration_seconds": "histogram",
    "smg_http_connections_active": "gauge",
    "smg_http_rate_limit_total": "counter",
    "smg_http_inflight_request_age_count": "gauge",
    "smg_router_requests_total": "counter",
    "smg_router_request_duration_seconds": "histogram",
    "smg_router_upstream_responses_total": "counter",
    "smg_router_request_errors_total": "counter",
    "smg_router_stage_duration_seconds": "histogram",
    "smg_router_ttft_seconds": "histogram",
    "smg_router_tpot_seconds": "histogram",
    "smg_router_tokens_total": "counter",
    "smg_router_generation_duration_seconds": "histogram",
    "smg_worker_pool_size": "gauge",
    "smg_worker_requests_active": "gauge",
    "smg_worker_routing_keys_active": "gauge",
    "smg_worker_connections_active": "gauge",
    "smg_worker_health": "gauge",
    "smg_worker_health_checks_total": "counter",
    "smg_worker_selection_total": "counter",
    "smg_worker_errors_total": "counter",
    "smg_manual_policy_cache_entries": "gauge",
    "smg_manual_policy_branch_total": "counter",
    "smg_consistent_hashing_policy_branch_total": "counter",
    "smg_prefix_hash_policy_branch_total": "counter",
    "smg_worker_cb_state": "gauge",
    "smg_worker_cb_transitions_total": "counter",
    "smg_worker_cb_outcomes_total": "counter",
    "smg_worker_cb_consecutive_failures": "gauge",
    "smg_worker_cb_consecutive_successes": "gauge",
    "smg_worker_retries_total": "counter",
    "smg_worker_retries_exhausted_total": "counter",
    "smg_worker_retry_backoff_seconds": "histogram",
    "smg_discovery_registrations_total": "counter",
    "smg_discovery_deregistrations_total": "counter",
    "smg_discovery_sync_duration_seconds": "histogram",
    "smg_discovery_workers_discovered": "gauge",
    "smg_mcp_tool_calls_total": "counter",
    "smg_mcp_tool_duration_seconds": "histogram",
    "smg_mcp_servers_active": "gauge",
    "smg_mcp_tool_iterations_total": "counter",
    "smg_db_operations_total": "counter",
    "smg_db_operation_duration_seconds": "histogram",
    "smg_db_connections_active": "gauge",
    "smg_db_items_stored": "counter",
}

NVIDIA_SMI_METRICS: dict[str, str] = {
    "utilization.gpu": "nvidia_smi:utilization_gpu",
    "utilization.memory": "nvidia_smi:utilization_memory",
    "memory.used": "nvidia_smi:memory_used_mib",
    "memory.total": "nvidia_smi:memory_total_mib",
    "temperature.gpu": "nvidia_smi:temperature_celsius",
    "power.draw": "nvidia_smi:power_usage_watts",
    "enforced.power.limit": "nvidia_smi:power_limit_watts",
    "clocks.sm": "nvidia_smi:clock_sm_mhz",
    "clocks_throttle_reasons.active": "nvidia_smi:clocks_throttle_reasons",
}


@dataclass(frozen=True)
class ScrapeTarget:
    source: Literal["engine", "router"]
    endpoint: str
    url: str

    @property
    def key(self) -> str:
        return f"{self.source}:{self.endpoint}"

    @property
    def metrics(self) -> dict[str, MetricKind]:
        return ENGINE_METRICS if self.source == "engine" else ROUTER_METRICS


class MetricSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    log_profile: str
    model_name: Optional[str]
    source: Literal["engine", "router", "nvidia_smi"]
    endpoint: str
    metric: str
    kind: MetricKind
    labels: dict[str, str]
    ts: float
    value: float
