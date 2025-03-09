from sqlalchemy import MetaData
from sqlalchemy.ext.automap import automap_base

from palzlib.database.db_client import SQLDBClient


class DatabaseMapper:
    """Factory to manage SQLAlchemy automapping."""

    _instance = None  # Singleton instance

    def __new__(cls, db_client: SQLDBClient = None):
        if cls._instance is None:
            if db_client is None:
                raise ValueError(
                    "A SQLDBClient instance must be provided for the first instantiation."
                )
            cls._instance = super(DatabaseMapper, cls).__new__(cls)
            cls._instance._initialize(db_client)
        return cls._instance

    def _initialize(self, db_client: SQLDBClient):
        """Initialize database mapping only once."""
        self.db_client = db_client
        self.metadata = MetaData()
        self.metadata.reflect(self.db_client.engine, views=True)
        self.base = automap_base(metadata=self.metadata)
        self.base.prepare(autoload_with=self.db_client.engine)
        print("db mapping")

    def get_model(self, table_name: str):
        """Returns the automapped model class for a given table."""
        if table_name not in self.base.classes:
            raise KeyError(f"Table '{table_name}' not found in database.")
        return self.base.classes[table_name]
