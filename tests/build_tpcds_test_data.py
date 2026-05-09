"""
Self-contained TPC-DS test data builder for SQLite.
Reads the OSI semantic model YAML directly — zero imports from app/.
Usage:
    uv run python tests/build_tpcds_test_data.py              # create tables + populate
    uv run python tests/build_tpcds_test_data.py --ddl-only   # create tables only
    uv run python tests/build_tpcds_test_data.py --drop-only  # drop all TPC-DS tables
    uv run python tests/build_tpcds_test_data.py --seed 123   # custom random seed
"""
import argparse
import asyncio
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import aiosqlite
import yaml

MODEL_PATH = Path(__file__).resolve().parent.parent / "config" / "semantics" / "tpcds_semantic_model.yaml"
TEST_MODEL_PATH = Path(__file__).resolve().parent / "config" / "semantics" / "tpcds_model_sqlite.yaml"
DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "test.db"


def infer_sqlite_type(col_name: str, expression: str) -> str:
    """Infer SQLite column type from column name and expression."""
    expr = expression.lower()
    if "||" in expr:
        return None  # computed — skip from DDL
    if any(op in expr for op in ("+", "-", "*", "/")) and not any(
        kw in expr for kw in ("_sk", "_id", "_key", "date", "name", "desc", "email")
    ):
        return "DECIMAL(10,2)"
    name = col_name.lower()
    # Surrogate keys are INTEGER, business keys (_id) are TEXT
    if name.endswith("_sk"):
        return "INTEGER"
    if name.endswith("_id"):
        return "TEXT"
    if "date" in name or "time" in name:
        return "DATE"
    if name == "d_year":
        return "INTEGER"
    if any(kw in name for kw in ("price", "amount", "cost", "profit", "sales")):
        return "DECIMAL(10,2)"
    if any(kw in name for kw in ("quantity", "count", "number", "employees", "ticket")):
        return "INTEGER"
    if any(kw in name for kw in ("name", "desc", "email", "city", "state", "brand", "category")):
        return "TEXT"
    return "TEXT"


def get_physical_columns(dataset: dict) -> list[tuple[str, str]]:
    """Extract physical column names and types from a dataset definition.
    Returns list of (col_name, sqlite_type). Skips computed columns (||, arithmetic).
    """
    columns: list[tuple[str, str]] = []
    field_names: set[str] = set()

    for field in dataset.get("fields", []):
        expr = field["expression"]["dialects"][0]["expression"]
        col_type = infer_sqlite_type(field["name"], expr)
        if col_type is not None:
            columns.append((field["name"], col_type))
            field_names.add(field["name"])

    pk = dataset.get("primary_key", [])
    if pk:
        for pk_col in pk:
            if pk_col not in field_names:
                dt = infer_sqlite_type(pk_col, pk_col)
                columns.append((pk_col, dt or "INTEGER"))

    return columns


def strip_schema(source: str) -> str:
    """Strip catalog/schema prefix for SQLite compatibility.
    'tpcds.public.store_sales' -> 'store_sales'
    """
    return source.rsplit(".", 1)[-1]


class TpcdsTestDataBuilder:
    def __init__(self, model_path: Path, seed: int = 42):
        with open(model_path, encoding="utf-8") as f:
            self._spec = yaml.safe_load(f)
        self._model = self._spec["semantic_model"][0]
        self._rng = random.Random(seed)
        self._datasets: dict[str, dict] = {ds["name"]: ds for ds in self._model["datasets"]}

    def get_sqlite_table_name(self, dataset_name: str) -> str:
        """Return SQLite-safe table name (schema prefix stripped)."""
        return strip_schema(self._datasets[dataset_name]["source"])

    def write_sqlite_model(self, output_path: Path):
        """Write a copy of the model YAML with schema-prefix-stripped source names.
        This model can be used with SqlExecTool for end-to-end testing.
        """
        import copy
        spec_copy = copy.deepcopy(self._spec)
        for ds in spec_copy["semantic_model"][0]["datasets"]:
            ds["source"] = strip_schema(ds["source"])
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(spec_copy, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

    def get_ddl(self) -> str:
        """Generate clean SQLite CREATE TABLE statements."""
        parts = []
        for ds_name, ds in self._datasets.items():
            table = self.get_sqlite_table_name(ds_name)
            cols = get_physical_columns(ds)
            col_defs = [f"    {name} {dtype}" for name, dtype in cols]

            pk = ds.get("primary_key", [])
            if pk:
                pk_names = [c for c in pk if any(rc[0] == c for rc in cols)]
                if pk_names:
                    col_defs.append(f"    PRIMARY KEY ({', '.join(pk_names)})")

            unique_keys = ds.get("unique_keys", [])
            for uk in unique_keys:
                uk_names = [c for c in uk if any(rc[0] == c for rc in cols)]
                if uk_names:
                    col_defs.append(f"    UNIQUE ({', '.join(uk_names)})")

            stmt = f"CREATE TABLE IF NOT EXISTS {table} (\n" + ",\n".join(col_defs) + "\n);"
            parts.append(stmt)
        return "\n\n".join(parts)

    async def create_tables(self, db_path: Path):
        async with aiosqlite.connect(str(db_path)) as conn:
            for ds_name in self._datasets:
                table = self.get_sqlite_table_name(ds_name)
                await conn.execute(f"DROP TABLE IF EXISTS {table}")
            ddl = self.get_ddl()
            await conn.executescript(ddl)
            await conn.commit()

    async def populate(self, db_path: Path):
        dates = self._generate_date_dim(count=60)
        customers = self._generate_customers(count=10)
        items = self._generate_items(count=10)
        stores = self._generate_stores(count=5)
        sales = self._generate_store_sales(count=100, dates=dates, customers=customers, items=items, stores=stores)

        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("PRAGMA foreign_keys = ON")
            await conn.commit()
            await self._insert_bulk(conn, "date_dim", dates)
            await self._insert_bulk(conn, "customer", customers)
            await self._insert_bulk(conn, "item", items)
            await self._insert_bulk(conn, "store", stores)
            await self._insert_bulk(conn, "store_sales", sales)
            await conn.commit()

    async def drop_tables(self, db_path: Path):
        async with aiosqlite.connect(str(db_path)) as conn:
            for ds_name in self._datasets:
                table = self.get_sqlite_table_name(ds_name)
                await conn.execute(f"DROP TABLE IF EXISTS {table}")
            await conn.commit()

    # ------------------------------------------------------------------ #
    #  Data generators
    # ------------------------------------------------------------------ #

    def _generate_date_dim(self, count: int = 60) -> list[dict]:
        base = date(2024, 1, 1)
        rows = []
        for i in range(count):
            d = base + timedelta(days=i)
            rows.append({
                "d_date_sk": i + 1,
                "d_date": d.isoformat(),
                "d_year": d.year,
                "d_quarter_name": f"{d.year}Q{(d.month - 1) // 3 + 1}",
                "d_month_name": d.strftime("%B"),
            })
        return rows

    def _generate_customers(self, count: int = 10) -> list[dict]:
        first_names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry", "Iris", "Jack"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
        rows = []
        for i in range(count):
            fn = first_names[i]
            ln = last_names[i]
            rows.append({
                "c_customer_sk": i + 1,
                "c_customer_id": f"CUST{1000 + i:05d}",
                "c_first_name": fn,
                "c_last_name": ln,
                "c_email_address": f"{fn.lower()}.{ln.lower()}@example.com",
            })
        return rows

    def _generate_items(self, count: int = 10) -> list[dict]:
        brands = ["BrandA", "BrandB", "BrandC"]
        categories = ["Electronics", "Clothing", "Home", "Sports", "Books"]
        rows = []
        for i in range(count):
            rows.append({
                "i_item_sk": i + 1,
                "i_item_id": f"ITEM{2000 + i:05d}",
                "i_item_desc": f"Product description for item {i + 1}",
                "i_brand": self._rng.choice(brands),
                "i_category": self._rng.choice(categories),
                "i_current_price": round(self._rng.uniform(5.0, 500.0), 2),
            })
        return rows

    def _generate_stores(self, count: int = 5) -> list[dict]:
        cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
        states = ["NY", "CA", "IL", "TX", "AZ"]
        rows = []
        for i in range(count):
            rows.append({
                "s_store_sk": i + 1,
                "s_store_id": f"STORE{3000 + i:05d}",
                "s_store_name": f"Store #{i + 1} - {cities[i]}",
                "s_city": cities[i],
                "s_state": states[i],
                "s_number_employees": self._rng.randint(20, 200),
            })
        return rows

    def _generate_store_sales(
        self, count: int, dates: list, customers: list, items: list, stores: list
    ) -> list[dict]:
        rows = []
        seen_pairs: set[tuple[int, int]] = set()
        items_pool = list(items)
        for i in range(count):
            date_row = self._rng.choice(dates)
            customer = self._rng.choice(customers)
            store = self._rng.choice(stores)
            quantity = self._rng.randint(1, 10)
            ticket = (i // 3) + 1

            # Ensure unique (item_sk, ticket_number)
            attempts = 0
            while attempts < 20:
                item = self._rng.choice(items_pool)
                if (ticket, item["i_item_sk"]) not in seen_pairs:
                    break
                attempts += 1
            else:
                continue  # skip if can't find unique pair
            seen_pairs.add((ticket, item["i_item_sk"]))

            price = item["i_current_price"]
            ext_price = round(quantity * price, 2)
            profit = round(ext_price * self._rng.uniform(0.05, 0.40), 2)
            rows.append({
                "ss_ticket_number": ticket,
                "ss_sold_date_sk": date_row["d_date_sk"],
                "ss_item_sk": item["i_item_sk"],
                "ss_customer_sk": customer["c_customer_sk"],
                "ss_store_sk": store["s_store_sk"],
                "ss_quantity": quantity,
                "ss_sales_price": price,
                "ss_ext_sales_price": ext_price,
                "ss_net_profit": profit,
            })
        return rows

    # ------------------------------------------------------------------ #
    #  Bulk insert helper
    # ------------------------------------------------------------------ #

    async def _insert_bulk(self, conn: aiosqlite.Connection, dataset_name: str, rows: list[dict]):
        if not rows:
            return
        table = self.get_sqlite_table_name(dataset_name)
        cols = list(rows[0].keys())
        placeholders = ", ".join("?" for _ in cols)
        values = [tuple(row[col] for col in cols) for row in rows]
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
        await conn.executemany(sql, values)


async def main():
    parser = argparse.ArgumentParser(description="Build TPC-DS test data for SQLite")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--ddl-only", action="store_true", help="Create tables only, skip data population")
    parser.add_argument("--drop-only", action="store_true", help="Drop all TPC-DS tables")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB), help="SQLite database path")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"Error: Model file not found: {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    builder = TpcdsTestDataBuilder(MODEL_PATH, seed=args.seed)
    db_path = Path(args.db)

    if args.drop_only:
        await builder.drop_tables(db_path)
        print("TPC-DS tables dropped.")
        return

    print(f"Database: {db_path}")
    print(f"Seed: {args.seed}")
    print()
    print("--- DDL Preview ---")
    print(builder.get_ddl())
    print()

    await builder.create_tables(db_path)
    print("Tables created.")

    if not args.ddl_only:
        await builder.populate(db_path)
        print("Data populated.")
        print("  date_dim:    60 rows")
        print("  customer:    10 rows")
        print("  item:        10 rows")
        print("  store:        5 rows")
        print("  store_sales: 100 rows")

    # Generate SQLite-compatible model YAML for use with SqlExecTool
    builder.write_sqlite_model(TEST_MODEL_PATH)
    print(f"\nSQLite-compatible model written to: {TEST_MODEL_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
