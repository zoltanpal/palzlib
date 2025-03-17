from typing import List

from sqlalchemy import MetaData
from sqlalchemy.ext.automap import (
    automap_base,
    generate_relationship,
    interfaces,
    name_for_collection_relationship,
)

from palzlib.database.db_client import DBClient


class DBMapping:
    def __init__(self, db_client: DBClient, mapping_tables: list = None):
        self.db_client = db_client
        self.mapping_tables = mapping_tables or []
        self.metadata = MetaData()
        self.db_classes = None
        self.sorted_tables: List[str] = []

        self._initialize_mapping()

    def _initialize_mapping(self):
        if not self.db_client:
            return

        reflection_options = {"views": True}
        if self.mapping_tables:
            reflection_options["only"] = self.mapping_tables

        self.metadata.reflect(self.db_client.engine, **reflection_options)

        AutoBase = automap_base(metadata=self.metadata)
        AutoBase.prepare(
            name_for_collection_relationship=self._name_for_collection_relationship,
            generate_relationship=self._generate_relationship,
        )

        self.sorted_tables = [table.name for table in self.metadata.sorted_tables]
        self.db_classes = AutoBase.classes

    def __getattr__(self, item: str):
        if self.db_classes and item in self.db_classes:
            return getattr(self.db_classes, item)
        raise AttributeError(f"Attribute '{item}' not found in mapped classes.")

    def __getitem__(self, item: str):
        if self.db_classes and item in self.db_classes:
            return self.db_classes[item]
        raise KeyError(f"Key '{item}' not found in mapped classes.")

    def get_model(self, item: str, default=None):
        return self.db_classes.get(item, default) if self.db_classes else default

    @property
    def mapped_table_names(self):
        return (
            [cls.__table__ for cls in self.db_classes.values()]
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
        return (
            constraint.name.lower()
            if constraint.name
            else name_for_collection_relationship(
                base, local_cls, referred_cls, constraint
            )
        )
