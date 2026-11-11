#!/usr/bin/env python3
"""CLI entry point for the canonical security identity migration inventory."""
from app.services.identity_migration import main


if __name__ == "__main__":
    raise SystemExit(main())
