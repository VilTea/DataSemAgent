"""Generate SQLite test data from the TPC-DS OSI semantic model.

Creates physical tables matching the model's source definitions, populates
them with realistic sample data, and respects primary/foreign key relationships.

Usage:
    uv run python tests/build_tpcds_test_data.py [--db data/test.db]
"""
import argparse
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path


# ── Configuration ──────────────────────────────────────────────────────────

CUSTOMERS = 10
ITEMS = 10
STORES = 5
SALES_ROWS = 500
DATE_COUNT = 365 * 3  # 3 years
START_DATE = date(2022, 1, 1)

FIRST_NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry", "Iris", "Jack"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
BRANDS = ["BrandA", "BrandB", "BrandC"]
CATEGORIES = ["Electronics", "Clothing", "Books"]
CITIES = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
STATES = ["NY", "CA", "IL", "TX", "AZ"]


def main():
    p = argparse.ArgumentParser(description="Build TPC-DS test database")
    p.add_argument("--db", default="data/test.db", help="SQLite database path")
    args = p.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    _create_tables(conn)
    _insert_date_dim(conn, DATE_COUNT)
    _insert_customers(conn)
    _insert_items(conn)
    _insert_stores(conn)
    _insert_store_sales(conn)
    _print_stats(conn)

    conn.commit()
    conn.close()
    print(f"\n  Database written to {db_path.resolve()}")


# ── DDL ────────────────────────────────────────────────────────────────────

def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE date_dim (
            d_date_sk    INTEGER PRIMARY KEY,
            d_date       TEXT    NOT NULL,
            d_year       INTEGER NOT NULL,
            d_quarter_name TEXT  NOT NULL,
            d_month_name TEXT    NOT NULL
        );

        CREATE TABLE customer (
            c_customer_sk   INTEGER PRIMARY KEY,
            c_customer_id   TEXT    NOT NULL UNIQUE,
            c_first_name    TEXT    NOT NULL,
            c_last_name     TEXT    NOT NULL,
            c_email_address TEXT    NOT NULL
        );

        CREATE TABLE item (
            i_item_sk       INTEGER PRIMARY KEY,
            i_item_id       TEXT    NOT NULL UNIQUE,
            i_item_desc     TEXT    NOT NULL,
            i_brand         TEXT    NOT NULL,
            i_category      TEXT    NOT NULL,
            i_current_price REAL    NOT NULL
        );

        CREATE TABLE store (
            s_store_sk          INTEGER PRIMARY KEY,
            s_store_id          TEXT    NOT NULL UNIQUE,
            s_store_name        TEXT    NOT NULL,
            s_city              TEXT    NOT NULL,
            s_state             TEXT    NOT NULL,
            s_number_employees  INTEGER NOT NULL
        );

        CREATE TABLE store_sales (
            ss_item_sk          INTEGER NOT NULL REFERENCES item(i_item_sk),
            ss_ticket_number    INTEGER NOT NULL,
            ss_sold_date_sk     INTEGER NOT NULL REFERENCES date_dim(d_date_sk),
            ss_customer_sk      INTEGER NOT NULL REFERENCES customer(c_customer_sk),
            ss_store_sk         INTEGER NOT NULL REFERENCES store(s_store_sk),
            ss_quantity         INTEGER NOT NULL,
            ss_sales_price      REAL    NOT NULL,
            ss_ext_sales_price  REAL    NOT NULL,
            ss_net_profit       REAL    NOT NULL,
            PRIMARY KEY (ss_item_sk, ss_ticket_number)
        );
    """)


# ── Data generation ────────────────────────────────────────────────────────

def _insert_date_dim(conn: sqlite3.Connection, count: int) -> None:
    rows = []
    for i in range(count):
        d = START_DATE + timedelta(days=i)
        year = d.year
        quarter = f"{year}Q{(d.month - 1) // 3 + 1}"
        month = d.strftime("%B")
        rows.append((i + 1, d.isoformat(), year, quarter, month))
    conn.executemany("INSERT INTO date_dim VALUES (?, ?, ?, ?, ?)", rows)


def _insert_customers(conn: sqlite3.Connection) -> None:
    rows = []
    for i in range(CUSTOMERS):
        sk = i + 1
        cid = f"CUST{1000 + i:05d}"
        first = FIRST_NAMES[i]
        last = LAST_NAMES[i]
        email = f"{first.lower()}.{last.lower()}@example.com"
        rows.append((sk, cid, first, last, email))
    conn.executemany("INSERT INTO customer VALUES (?, ?, ?, ?, ?)", rows)


def _insert_items(conn: sqlite3.Connection) -> None:
    random.seed(42)
    rows = []
    for i in range(ITEMS):
        sk = i + 1
        iid = f"ITEM{2000 + i:05d}"
        desc = f"Product description for item {sk}"
        brand = random.choice(BRANDS)
        category = random.choice(CATEGORIES)
        price = round(random.uniform(10, 400), 2)
        rows.append((sk, iid, desc, brand, category, price))
    conn.executemany("INSERT INTO item VALUES (?, ?, ?, ?, ?, ?)", rows)


def _insert_stores(conn: sqlite3.Connection) -> None:
    random.seed(99)
    rows = []
    for i in range(STORES):
        sk = i + 1
        sid = f"STORE{3000 + i:05d}"
        name = f"Store #{sk} - {CITIES[i]}"
        city = CITIES[i]
        state = STATES[i]
        emp = random.randint(50, 120)
        rows.append((sk, sid, name, city, state, emp))
    conn.executemany("INSERT INTO store VALUES (?, ?, ?, ?, ?, ?)", rows)


def _insert_store_sales(conn: sqlite3.Connection) -> None:
    random.seed(7)
    ticket = 0
    rows = []
    for _ in range(SALES_ROWS):
        item_sk = random.randint(1, ITEMS)
        store_sk = random.randint(1, STORES)
        customer_sk = random.randint(1, CUSTOMERS)
        date_sk = random.randint(1, DATE_COUNT)

        cur = conn.execute(
            "SELECT i_current_price FROM item WHERE i_item_sk = ?", (item_sk,)
        )
        base_price = cur.fetchone()[0]
        quantity = random.randint(1, 10)
        sales_price = round(base_price * random.uniform(0.5, 1.2), 2)
        ext_price = round(sales_price * quantity, 2)
        net_profit = round(ext_price * random.uniform(0.05, 0.40), 2)

        ticket += 1
        rows.append((
            item_sk, ticket, date_sk, customer_sk, store_sk,
            quantity, sales_price, ext_price, net_profit,
        ))

    conn.executemany(
        "INSERT INTO store_sales VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
    )


def _print_stats(conn: sqlite3.Connection) -> None:
    for table in ("date_dim", "customer", "item", "store", "store_sales"):
        cnt = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {cnt} rows")


if __name__ == "__main__":
    main()
