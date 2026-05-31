"""Tests for EvalCollector and redaction utility."""
from app.eval.collector import redact, _is_sensitive_key


class TestRedaction:
    SENSITIVE = frozenset({"api_key", "password", "token", "secret", "authorization"})

    def test_exact_key_match_redacted(self):
        obj = {"api_key": "sk-abc123", "name": "test"}
        result = redact(obj, self.SENSITIVE)
        assert result["api_key"] == "***"
        assert result["name"] == "test"

    def test_segmented_key_match_redacted(self):
        obj = {"x-api-key": "sk-abc123"}
        result = redact(obj, self.SENSITIVE)
        assert result["x-api-key"] == "***"

    def test_partial_match_not_redacted(self):
        obj = {"token_count": 100, "authorization_header": "Bearer xyz"}
        result = redact(obj, self.SENSITIVE)
        assert result["token_count"] == 100
        assert result["authorization_header"] == "Bearer xyz"

    def test_value_pattern_redacted(self):
        obj = {"headers": "Authorization: Bearer sk-abc123"}
        result = redact(obj, self.SENSITIVE)
        assert "***" in result["headers"]
        assert "sk-abc123" not in result["headers"]

    def test_nested_dict_redacted(self):
        obj = {"config": {"api_key": "sk-abc123", "timeout": 30}}
        result = redact(obj, self.SENSITIVE)
        assert result["config"]["api_key"] == "***"
        assert result["config"]["timeout"] == 30

    def test_list_of_dicts_redacted(self):
        obj = [{"name": "a", "password": "pwd1"}, {"name": "b"}]
        result = redact(obj, self.SENSITIVE)
        assert result[0]["password"] == "***"

    def test_primitive_passthrough(self):
        assert redact("hello", self.SENSITIVE) == "hello"
        assert redact(42, self.SENSITIVE) == 42
        assert redact(None, self.SENSITIVE) is None


class TestSensitiveKey:
    SENSITIVE = frozenset({"api_key", "token"})

    def test_exact_match(self):
        assert _is_sensitive_key("api_key", self.SENSITIVE)

    def test_case_insensitive(self):
        assert _is_sensitive_key("API_KEY", self.SENSITIVE)

    def test_segment_match(self):
        assert _is_sensitive_key("x-api-key", self.SENSITIVE)
        assert _is_sensitive_key("auth.token.value", self.SENSITIVE)

    def test_partial_no_match(self):
        assert not _is_sensitive_key("token_count", self.SENSITIVE)
        assert not _is_sensitive_key("api_key_v2", self.SENSITIVE)
