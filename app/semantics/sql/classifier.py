from app.semantics.sql.exceptions import FieldNotFoundError, MetricNotFoundError


class FieldClassifier:
    """Classifies SQL expression items into semantic field types.

    Determines whether a SELECT expression references a metric, dimension,
    outer reference, or plain column based on the OSI semantic model.
    """

    def __init__(self, parser, strict: bool = True):
        self._parser = parser
        self._strict = strict
        self._known_datasets = {ds.name for ds in parser._model.datasets}

    @staticmethod
    def get_item_name(item) -> str:
        """Extract the logical name from an AST item."""
        import sqlglot.expressions as exp
        if isinstance(item, exp.Column):
            return item.name
        if isinstance(item, exp.Alias):
            inner = item.this
            # COUNT(*) AS alias — no column name to resolve
            if isinstance(inner, exp.AggFunc):
                for child in inner.walk():
                    if isinstance(child, exp.Star):
                        return ""
            return inner.name if hasattr(inner, 'name') else str(inner)
        if isinstance(item, exp.AggFunc):
            for child in item.walk():
                if isinstance(child, exp.Star):
                    return ""  # COUNT(*) — no column name to resolve
        if hasattr(item, "this") and isinstance(item.this, exp.Column):
            return item.this.name
        if hasattr(item, "name"):
            return item.name
        return str(item)

    @staticmethod
    def get_item_alias(item) -> str | None:
        """Extract the alias from an AST item if present."""
        import sqlglot.expressions as exp
        if isinstance(item, exp.Alias):
            return item.alias
        return None

    def classify(self, item, alias: str, ctx, scope_mgr):
        """Classify a SELECT expression item and return (field_type, col_info).

        Returns:
            tuple of (field_type: str, col_info: None)
            field_type is one of: "metric", "dimension", "outer_ref", "column"
        """
        import sqlglot.expressions as exp

        name = self.get_item_name(item)

        if not alias and not name:
            return "column", None

        if isinstance(item, (exp.Literal, exp.Star)):
            return "column", None

        # COUNT(*) / COUNT(1) — pure row counting, no field reference
        if isinstance(item, exp.AggFunc):
            for child in item.walk():
                if isinstance(child, exp.Star):
                    return "column", None

        if isinstance(item, exp.Column) and item.table:
            if item.table not in self._known_datasets:
                if item.table in ctx.table_alias_map:
                    resolved = ctx.table_alias_map[item.table]
                    known_sources = {ds.source for ds in self._parser._model.datasets}
                    if resolved not in known_sources and resolved not in self._known_datasets:
                        # True CTE/subquery alias — pass through, don't resolve as OSI field
                        return "column", None
                    # Dataset alias (e.g. 'c' for 'customer') — fall through to OSI checks
                else:
                    in_scope = scope_mgr.is_table_in_current_scope(ctx, item.table)
                    if not in_scope:
                        return "outer_ref", None

        if name and self._parser.is_metric(name):
            return "metric", None
        if alias and self._parser.is_metric(alias):
            return "metric", None

        if name and self._parser.is_dimension(name):
            return "dimension", None
        if alias and self._parser.is_dimension(alias):
            return "dimension", None

        outer_ref = scope_mgr.resolve_column(ctx, alias)
        if outer_ref:
            return "outer_ref", outer_ref

        # CTE / derived table references are already translated — pass through
        if isinstance(item, exp.Column) and item.table:
            known_sources = {ds.source for ds in self._parser._model.datasets}
            resolved = ctx.table_alias_map.get(item.table, item.table)
            if item.table not in self._known_datasets and resolved not in known_sources:
                return "column", None

        field_name = name if name and name != alias else alias
        if field_name:
            try:
                mapping = self._parser.resolve_field(field_name)
                if mapping.is_dimension:
                    return "dimension", None
                if mapping.is_metric:
                    return "metric", None
                return "column", None
            except FieldNotFoundError:
                return "column", None  # let translator handle strict validation

        return "column", None

    def _is_from_subquery_scope(self, ctx, scope_mgr) -> bool:
        """True if the current scope contains CTE or derived table aliases."""
        for alias in ctx.table_alias_map:
            if alias not in self._known_datasets:
                return True
        return False
