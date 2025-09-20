import os
import pytest
import jwt
from src.core.config import SETTINGS


def make_token(sub='test-user', role='user'):
    secret = getattr(SETTINGS, 'jwt_secret', 'testsecret') or 'testsecret'
    payload = {'sub': sub, 'role': role}
    return jwt.encode(payload, secret, algorithm='HS256')


def test_require_roles_enforced(gateway_test_client, monkeypatch):
    # Ensure role-based auth is enabled
    monkeypatch.setenv('ENABLE_ROLE_BASED_AUTH', 'true')
    SETTINGS.enable_role_based_auth = True

    token = make_token(role='user')
    headers = {'Authorization': f'Bearer {token}'}
    # call an admin-only endpoint (anonymize)
    resp = gateway_test_client.post('/api/v1/anonymize', json={'text': 'Emily Dawson is here'})
    # Without token expect 401
    assert resp.status_code in (401, 403)

    # With user token still should be forbidden
    resp2 = gateway_test_client.post('/api/v1/anonymize', json={'text': 'Emily Dawson is here'}, headers=headers)
    assert resp2.status_code in (401, 403)


def test_require_roles_bypassed_when_disabled(gateway_test_client, monkeypatch):
    monkeypatch.setenv('ENABLE_ROLE_BASED_AUTH', 'false')
    SETTINGS.enable_role_based_auth = False

    # Should be allowed without token when flag disabled
    resp = gateway_test_client.post('/api/v1/anonymize', json={'text': 'Emily Dawson is here'})
    assert resp.status_code == 200
