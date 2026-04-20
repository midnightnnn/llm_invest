"""Provider registry and transport metadata."""

from arena.providers.anthropic_patches import apply_anthropic_effort_patch
from arena.providers.credentials import (
    apply_model_secret_payload,
    build_model_secret_payload,
    normalize_provider_token,
    parse_model_secret_providers,
    runtime_credential_flags,
    runtime_row_api_key_status,
)
from arena.providers.registry import (
    ProviderSpec,
    canonical_provider,
    default_model_for_provider,
    get_provider_spec,
    list_adk_provider_specs,
    list_helper_provider_specs,
    list_provider_specs,
    provider_api_key_from_settings,
    provider_alias_map,
    provider_base_url_from_settings,
    provider_has_credentials,
    provider_has_direct_api_key,
)

apply_anthropic_effort_patch()


__all__ = [
    "apply_anthropic_effort_patch",
    "apply_model_secret_payload",
    "build_model_secret_payload",
    "normalize_provider_token",
    "parse_model_secret_providers",
    "ProviderSpec",
    "canonical_provider",
    "default_model_for_provider",
    "get_provider_spec",
    "list_adk_provider_specs",
    "list_helper_provider_specs",
    "list_provider_specs",
    "provider_api_key_from_settings",
    "provider_alias_map",
    "provider_base_url_from_settings",
    "provider_has_credentials",
    "provider_has_direct_api_key",
    "runtime_credential_flags",
    "runtime_row_api_key_status",
]
