"""
The DbMapping class uses SQLAlchemy's automap feature to map database tables to Python classes.
It takes a DbClient and an optional list of table names.
After reflection, it provides access to mapped classes via attribute, item, or get method lookups.
"""

from sqlalchemy import MetaData
from sqlalchemy.ext.automap import (
    automap_base,
    generate_relationship,
    interfaces,
    name_for_collection_relationship,
)

from palzlib.database.db_client import DBClient

class DbMapping:
    def __init__(self, db_client: DbClient, mapping_tables: list = None):
        self.metadata = MetaData()
        self.sorted_tables = []
        self.db_client = db_client
        self.mapping_tables = mapping_tables
        self.db_classes = None

        self.mapping()

    def mapping(self):
        if not self.db_client:
            return False

        if self.mapping_tables:
            self.metadata.reflect(
                self.db_client.engine, only=self.mapping_tables, views=True
            )
        else:
            self.metadata.reflect(self.db_client.engine, views=True)

        AutoBase = automap_base(metadata=self.metadata)
        AutoBase.prepare(
            name_for_collection_relationship=self._name_for_collection_relationship,
            generate_relationship=self._generate_relationship,
        )

        self.sorted_tables = [table.name for table in self.metadata.sorted_tables]
        self.db_classes = AutoBase.classes

    def __getattr__(self, item: str):
        if item not in self.db_classes:
            raise AttributeError(f"Attribute '{item}' not found in mapped classes.")
        return getattr(self.db_classes, item)

    def __getitem__(self, item: str):
        if item not in self.db_classes:
            raise KeyError(f"Key '{item}' not found in mapped classes.")
        return self.db_classes[item]

    def get(self, item: str, default=None):
        return self.db_classes.get(item, default)

    @property
    def mapped_table_names(self):
        return (
            [x.__table__ for x in list(self.db_classes.db_classes)]
            if self.db_classes
            else []
        )

    @staticmethod
    def _generate_relationship(
        base, direction, return_fn, attr_name, local_cls, referred_cls, **kw
    ):
        if direction is interfaces.ONETOMANY:
            kw["cascade"] = "all, delete"
            kw["passive_deletes"] = False

        return generate_relationship(
            base, direction, return_fn, attr_name, local_cls, referred_cls, **kw
        )

    @staticmethod
    def _name_for_collection_relationship(base, local_cls, referred_cls, constraint):
        if constraint.name:
            return constraint.name.lower()

        return name_for_collection_relationship(
            base, local_cls, referred_cls, constraint
        )
