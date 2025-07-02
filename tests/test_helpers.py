import time
from appv2 import estimate_tokens, validate_token_limit, truncate_response, RateLimiter


def test_estimate_tokens_basic():
    text = "hello world"
    assert estimate_tokens(text) > 0


def test_validate_token_limit():
    assert validate_token_limit("short text")
    long_text = "x" * (500 * 4 + 10)
    assert not validate_token_limit(long_text)


def test_truncate_response():
    long_text = "x" * (500 * 4 + 10)
    truncated = truncate_response(long_text)
    assert "truncated" in truncated.lower()


def test_rate_limiter_calls_trimmed():
    limiter = RateLimiter(2)
    limiter.calls = [time.time() - 61, time.time() - 30]
    limiter.check_rate_limit()
    assert len(limiter.calls) <= 2
