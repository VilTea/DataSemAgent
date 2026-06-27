class TokenCounter:
    """Lazy-loads tiktoken cl100k_base encoding, counts tokens."""

    def __init__(self):
        self._enc = None

    def count(self, obj: str | list[dict]) -> int:
        if self._enc is None:
            import tiktoken
            self._enc = tiktoken.get_encoding("cl100k_base")
        if isinstance(obj, str):
            return len(self._enc.encode(obj))
        # list of chat message dicts: role + " " + content
        total = 0
        for m in obj:
            total += len(self._enc.encode(
                (m.get("role") or "") + " " + (m.get("content") or "")
            ))
        return total
