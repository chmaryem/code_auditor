"""Add github_token column to users table.

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-18 00:00:00
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("github_token", sa.String(500), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "github_token")
