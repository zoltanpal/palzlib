import pytest

from palzlib.database.db_config import DBConfig


def test_valid_dbconfig():
    config = DBConfig(
        username="user",
        password="pass",
        dbname="database",
        host="localhost",
    )
    assert config.username == "user"
    assert config.password == "pass"
    assert config.dbname == "database"
    assert config.host == "localhost"
    assert config.dialect == "postgresql+psycopg2"
    assert config.port == 5432


def test_default_values():
    config = DBConfig(
        username="user", password="pass", dbname="database", host="localhost"
    )
    assert config.dialect == "postgresql+psycopg2"
    assert config.port == 5432


def test_missing_required_fields():
    with pytest.raises(TypeError):
        DBConfig()

    with pytest.raises(TypeError):
        DBConfig(username="user", password="pass")

    with pytest.raises(ValueError):
        DBConfig(username="", password="pass", dbname="database", host="localhost")


@pytest.mark.parametrize(
    "config",
    [
        {
            "username": 123,
            "password": "pass",
            "dbname": "database",
            "host": "localhost",
        },
        {
            "username": "user",
            "password": 657,
            "dbname": "database",
            "host": "localhost",
        },
        {"username": "user", "password": "pass", "dbname": "database", "host": 5344},
    ],
)
def test_invalid_field_types(config):
    with pytest.raises(TypeError):
        DBConfig(**config)
