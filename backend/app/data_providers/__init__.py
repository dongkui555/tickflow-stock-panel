"""Market data provider abstraction.

Providers normalize external data sources into the internal parquet schema.
"""
from app.data_providers.base import AssetType, MarketDataProvider, ProviderCapabilities


def get_provider(name: str = "tickflow"):
    """Load the online provider registry only when a provider is requested."""
    from app.data_providers.registry import get_provider as resolve_provider

    return resolve_provider(name)

__all__ = ["AssetType", "MarketDataProvider", "ProviderCapabilities", "get_provider"]
