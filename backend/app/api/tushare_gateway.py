"""Safe access to the configured Tushare-compatible gateway catalog."""
from __future__ import annotations

import math
import os
import re
from datetime import date, datetime
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.plugins.tushare.provider import (
    TushareAPIFailover,
    TushareProvider,
    build_datahub_client,
)
from app.services.tushare_catalog import TUSHARE_API_CATALOG, catalog_payload
from app.services.tushare_ext_sync import DEFAULT_AUTO_APIS, EXT_SPECS

router = APIRouter(prefix="/api/tushare", tags=["tushare"])

_PARAM_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")
_SENSITIVE_PARAM_TOKENS = ("token", "secret", "password", "api_key", "apikey")


class GatewayQueryRequest(BaseModel):
    api: str
    source: Literal["auto", "promax", "datahub"] = "auto"
    params: dict[str, Any] = Field(default_factory=dict)
    max_rows: int = Field(default=200, ge=1, le=5000)


def _env_enabled(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _configured_status() -> dict[str, bool]:
    promax_url = os.getenv("TUSHARE_PROMAX_URL", "").strip() or settings.tushare_promax_url.strip()
    promax_key = os.getenv("TUSHARE_PROMAX_KEY", "").strip() or settings.tushare_promax_key.strip()
    datahub_url = os.getenv("TUSHARE_DATAHUB_URL", "").strip() or settings.tushare_datahub_url.strip()
    datahub_key = (
        os.getenv("TUSHARE_DATAHUB_KEY", "").strip()
        or settings.tushare_datahub_key.strip()
        or promax_key
    )
    return {
        "promax": bool(promax_url and promax_key),
        "datahub": _env_enabled("TUSHARE_DATAHUB_ENABLED", settings.tushare_datahub_enabled)
        and bool(datahub_url and datahub_key),
    }


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in params.items():
        if not _PARAM_RE.fullmatch(key) or any(token in key.lower() for token in _SENSITIVE_PARAM_TOKENS):
            raise HTTPException(400, f"不允许的参数名: {key}")
        if value is None or value == "":
            continue
        if isinstance(value, list):
            if not all(isinstance(item, (str, int, float, bool)) for item in value):
                raise HTTPException(400, f"参数 {key} 只允许标量或标量数组")
            cleaned[key] = ",".join(str(item) for item in value)
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            raise HTTPException(400, f"参数 {key} 类型不受支持")
    return cleaned


def _build_client(source: str):
    status = _configured_status()
    if source == "datahub":
        if not status["datahub"]:
            raise HTTPException(503, "datahubco 网关未配置")
        return build_datahub_client(), "datahub"

    provider = TushareProvider()
    if source == "promax":
        if not status["promax"]:
            raise HTTPException(503, "Promax 网关未配置")
        return provider._promax_client(), "promax"  # noqa: SLF001

    try:
        primary = provider._ensure_api()  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001
        if not status["datahub"]:
            raise HTTPException(503, "没有可用的 Tushare 网关") from exc
        return build_datahub_client(), "datahub"
    if isinstance(primary, TushareAPIFailover):
        return primary, "auto"
    if not status["datahub"]:
        return primary, "promax" if provider._connection_mode() == "promax" else "official"  # noqa: SLF001
    primary_name = "promax" if provider._connection_mode() == "promax" else "official"  # noqa: SLF001
    return TushareAPIFailover(primary, build_datahub_client(), primary_name, "datahub"), "auto"


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


@router.get("/catalog")
def catalog() -> dict:
    status = _configured_status()
    items = catalog_payload()
    for item in items:
        item["integrated"] = item["http_supported"] and (status["promax"] or status["datahub"])
        item["auto_sync"] = item["api"] in DEFAULT_AUTO_APIS
        item["structured_storage"] = item["api"] in EXT_SPECS
    return {
        "count": len(items),
        "items": items,
        "sources": status,
        "source_chain": os.getenv("TUSHARE_EXT_SOURCE_CHAIN", "").strip()
        or settings.tushare_ext_source_chain,
        "auction_source": "promax",
    }


@router.post("/query")
def query_gateway(body: GatewayQueryRequest) -> dict:
    item = TUSHARE_API_CATALOG.get(body.api)
    if item is None:
        raise HTTPException(404, "接口不在80接口白名单中")
    if not item.http_supported:
        raise HTTPException(400, item.note or "该接口不支持 HTTP 调用")
    client, source_label = _build_client(body.source)
    params = _clean_params(body.params)
    try:
        frame = getattr(client, body.api)(**params)
        if frame is None:
            frame = pd.DataFrame()
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        # _PromaxTushareAPI uses __getattr__ for API methods, so getattr on an
        # unknown metadata attribute would return a functools.partial rather
        # than None. Read the instance dictionary explicitly.
        actual_source = client.__dict__.get("last_source") or source_label
        preview = [
            {key: _json_value(value) for key, value in row.items()}
            for row in frame.head(body.max_rows).to_dict(orient="records")
        ]
        return {
            "api": body.api,
            "label": item.label,
            "source": actual_source,
            "row_count": len(frame),
            "returned_rows": len(preview),
            "truncated": len(frame) > len(preview),
            "fields": list(frame.columns),
            "items": preview,
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"{item.label} 调用失败，已按主备链路尝试") from exc
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
