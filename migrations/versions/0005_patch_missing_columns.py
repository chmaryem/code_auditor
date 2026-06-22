"""Safely add columns that 0004 may have skipped (IF NOT EXISTS guard).

Adds to projects:  github_repo_url, active_branch, last_activity_at
Adds to conversations: deleted_at

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-18
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.text("""
        ALTER TABLE projects
            ADD COLUMN IF NOT EXISTS github_repo_url  VARCHAR(500),
            ADD COLUMN IF NOT EXISTS active_branch    VARCHAR(300),
            ADD COLUMN IF NOT EXISTS last_activity_at TIMESTAMP WITH TIME ZONE
    """))
    conn.execute(sa.text("""
        ALTER TABLE conversations
            ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP WITH TIME ZONE
    """))


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.text("ALTER TABLE projects DROP COLUMN IF EXISTS github_repo_url"))
    conn.execute(sa.text("ALTER TABLE projects DROP COLUMN IF EXISTS active_branch"))
    conn.execute(sa.text("ALTER TABLE projects DROP COLUMN IF EXISTS last_activity_at"))
    conn.execute(sa.text("ALTER TABLE conversations DROP COLUMN IF EXISTS deleted_at"))
