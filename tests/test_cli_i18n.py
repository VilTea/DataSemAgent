# tests/test_cli_i18n.py
import json
import pytest
from pathlib import Path


class TestI18nLoader:
    @pytest.fixture
    def i18n_dir(self, tmp_path):
        zh = tmp_path / "zh.json"
        zh.write_text(
            json.dumps({"greeting": "你好", "farewell": "再见"}, ensure_ascii=False), "utf-8"
        )
        en = tmp_path / "en.json"
        en.write_text(json.dumps({"greeting": "Hello", "farewell": "Goodbye"}), "utf-8")
        return tmp_path

    def test_load_zh(self, i18n_dir, monkeypatch):
        monkeypatch.setattr("app.cli.i18n.PROJECT_ROOT", i18n_dir.parent)
        from app.cli.i18n import I18nLoader
        loader = I18nLoader(lang="zh")
        loader._load(i18n_dir / "zh.json")
        assert loader.t("greeting") == "你好"

    def test_load_en(self, i18n_dir, monkeypatch):
        monkeypatch.setattr("app.cli.i18n.PROJECT_ROOT", i18n_dir.parent)
        from app.cli.i18n import I18nLoader
        loader = I18nLoader(lang="en")
        loader._load(i18n_dir / "en.json")
        assert loader.t("greeting") == "Hello"

    def test_fallback_to_en_when_lang_missing(self, i18n_dir, monkeypatch):
        monkeypatch.setattr("app.cli.i18n.PROJECT_ROOT", i18n_dir.parent)
        from app.cli.i18n import I18nLoader
        loader = I18nLoader(lang="fr")
        loader._load(i18n_dir / "en.json")
        assert loader.t("greeting") == "Hello"

    def test_missing_key_returns_key_itself(self, i18n_dir, monkeypatch):
        monkeypatch.setattr("app.cli.i18n.PROJECT_ROOT", i18n_dir.parent)
        from app.cli.i18n import I18nLoader
        loader = I18nLoader(lang="zh")
        loader._load(i18n_dir / "zh.json")
        assert loader.t("nonexistent_key") == "nonexistent_key"

    def test_format_kwargs(self, i18n_dir, monkeypatch):
        monkeypatch.setattr("app.cli.i18n.PROJECT_ROOT", i18n_dir.parent)
        from app.cli.i18n import I18nLoader
        zh_path = i18n_dir / "zh.json"
        zh_path.write_text(
            json.dumps({"items": "共 {count} 项"}, ensure_ascii=False), "utf-8"
        )
        loader = I18nLoader(lang="zh")
        loader._load(zh_path)
        assert loader.t("items", count=42) == "共 42 项"
