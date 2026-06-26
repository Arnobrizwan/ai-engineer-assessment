"""Idempotent entry point to create and seed ``analytics.db``.

Usage:
    python seed.py

This (re)creates the demo e-commerce schema and deterministic sample data at
the database path configured by ``DATABASE_URL`` in your environment / ``.env``.
"""

from __future__ import annotations

from agent.config import get_settings
from agent.db import seed


def main() -> None:
    """Seed the configured database and print a short confirmation."""
    settings = get_settings()
    path = seed(settings.db_path)
    print(f"Seeded analytics database at: {path}")
    print("Tables: customers, products, orders, order_items")


if __name__ == "__main__":
    main()
