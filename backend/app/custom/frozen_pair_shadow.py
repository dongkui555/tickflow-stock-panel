"""Read-only status for the independently scheduled frozen research pair."""
from fastapi import APIRouter, Request

from app.extensions import BACKEND_EXTENSION_API_VERSION, BackendExtensionRegistrar

EXTENSION_ID = 'research.frozen-pair-shadow'
EXTENSION_API_VERSION = BACKEND_EXTENSION_API_VERSION
router = APIRouter(prefix='/api/stockformer')


@router.get('/frozen-pair-shadow')
def frozen_pair_shadow_status(request: Request):
    from app.services.stockformer_pair_shadow import status
    return status(request.app.state.repo.store.data_dir)


@router.get('/decision-pair-shadow')
def decision_pair_shadow_status(request: Request):
    from app.services.stockformer_pair_shadow import status
    return status(request.app.state.repo.store.data_dir, variant='decision')


@router.get('/repaired-pair-shadow')
def repaired_pair_shadow_status(request: Request):
    from app.services.stockformer_pair_shadow import status
    return status(request.app.state.repo.store.data_dir, variant='repaired')


def setup(registrar: BackendExtensionRegistrar) -> None:
    registrar.include_router(router)
