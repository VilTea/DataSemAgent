# app/cli/i18n.py
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class I18nLoader:
    def __init__(self, lang: str = "zh"):
        self._messages: dict[str, str] = {}
        path = PROJECT_ROOT / "config" / "i18n" / f"{lang}.json"
        if not path.exists():
            path = PROJECT_ROOT / "config" / "i18n" / "en.json"
        if path.exists():
            self._load(path)

    def _load(self, path: Path) -> None:
        self._messages: dict[str, str] = json.loads(path.read_text("utf-8"))

    def t(self, key: str, **kwargs) -> str:
        msg = self._messages.get(key, key)
        return msg.format(**kwargs) if kwargs else msg
