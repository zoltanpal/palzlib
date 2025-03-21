from unittest import mock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from palzlib.database.db_client import DBClient


def test_dbclient_initialization(mock_db_client):
    assert mock_db_client.connection_string.startswith(
        "postgresql+psycopg2://test_user:test_pass@localhost:5432/test_db"
    )
    assert mock_db_client.engine is not None


def test_dbclient_session(mock_db_client):
    with mock_db_client.get_db_session() as session:
        assert session is not None
        assert session.bind == mock_db_client.engine


def test_invalid_dbconfig():
    with pytest.raises(ValueError):
        DBClient(None)


def test_engine_creation_failure(mock_db_config):
    mock.patch(
        "sqlalchemy.create_engine",
        side_effect=SQLAlchemyError("Engine creation failed"),
    )
    with pytest.raises(
        SQLAlchemyError, match="Database engine creation failed: Engine creation failed"
    ):
        DBClient(mock_db_config)
