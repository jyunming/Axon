import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_config_validate_route(live_api_server, make_brain):
    make_brain()
    base_url = live_api_server()

    async with AsyncClient(base_url=base_url) as client:
        res = await client.get("/config/validate")
        assert res.status_code == 200
        data = res.json()
        assert "issues" in data
