from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.config import Settings


def test_auth_defaults_are_dev_friendly():
    s = Settings()
    assert s.auth_enabled is True
    assert s.app_env == "dev"
    assert s.jwt_alg == "HS256"
    assert s.auth_dev_signer_enabled is False
    assert s.max_question_chars == 8000
    assert s.max_acl_tags == 32


def test_prod_requires_hs256_secret():
    with pytest.raises(ValidationError):
        Settings(app_env="prod", jwt_alg="HS256", jwt_secret="",
                 jwt_issuer="iss", jwt_audience="aud")


def test_prod_requires_issuer_and_audience():
    with pytest.raises(ValidationError):
        Settings(app_env="prod", jwt_alg="HS256", jwt_secret="s",
                 jwt_issuer="", jwt_audience="")


def test_prod_forbids_dev_signer():
    with pytest.raises(ValidationError):
        Settings(app_env="prod", jwt_alg="HS256", jwt_secret="s",
                 jwt_issuer="iss", jwt_audience="aud", auth_dev_signer_enabled=True)


def test_prod_valid_config_constructs():
    s = Settings(app_env="prod", jwt_alg="RS256", jwks_url="https://idp/jwks",
                 jwt_issuer="iss", jwt_audience="aud")
    assert s.app_env == "prod"
