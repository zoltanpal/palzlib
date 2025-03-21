from unittest.mock import MagicMock

import pytest


@pytest.fixture(scope="module")
def mock_db_client():
    """Mock the database client."""
    return MagicMock()


@pytest.fixture(scope="module")
def mock_db_mapper(mock_db_client):
    """Mock the database mapping."""
    db_mapper = MagicMock()
    db_mapper.db_classes.users = MagicMock()  # Mock User model
    return db_mapper
