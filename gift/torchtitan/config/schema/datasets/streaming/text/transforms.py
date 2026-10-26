from torchtitan.config.errors import ConfigFieldError

DEFAULT_FIM_RATES: dict[str, float] = {
    "vanilla_fim": 0.0,
    "inline_fim": 0.0,
    "multiline_fim": 0.0,
    "syntax_aware_fim": 0.0,
}


def validate_fim_rates(rates: dict[str, float] | None) -> None:
    unknown = set(rates or {}) - DEFAULT_FIM_RATES.keys()
    if unknown:
        raise ConfigFieldError("fim_rates", f"unknown FIM variants: {sorted(unknown)}")
