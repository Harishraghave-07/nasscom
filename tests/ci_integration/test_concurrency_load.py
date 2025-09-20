import asyncio
import pytest


@pytest.mark.asyncio
async def test_concurrent_detection_requests(gateway_test_client, mock_fast_analyzer):
    # Use the test client's base URL to run async httpx requests
    from httpx import AsyncClient
    base = gateway_test_client.base_url

    async def do_req(i):
        async with AsyncClient(base_url=str(base)) as ac:
            resp = await ac.post('/api/v1/detect', json={'text': f'Emily Dawson {i}'}, timeout=10.0)
            return resp.status_code

    tasks = [do_req(i) for i in range(20)]
    results = await asyncio.gather(*tasks)
    assert all(r == 200 for r in results)
