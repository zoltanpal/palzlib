import pytest

from palzlib.database.db_client import DBClient
from palzlib.database.db_config import DBConfig


@pytest.fixture(scope="session")
def mock_db_config():
    return DBConfig(
        username="test_user", password="test_pass", dbname="test_db", host="localhost"
    )


@pytest.fixture(scope="session")
def mock_db_client(mock_db_config):
    return DBClient(mock_db_config)
