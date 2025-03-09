from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from palzlib.database.db_config import DBConfig


class SQLDBClient:
    """
    Database client for managing SQLAlchemy sessions and connections.

    Attributes:
        connection_string (str): The connection string for the database.
        auto_commit (bool): Whether to enable autocommit.
        auto_flush (bool): Whether to enable autoflush.
        expire_on_commit (bool): Whether to expire objects on commit.
        engine (Engine): SQLAlchemy database engine.
        session_local (sessionmaker): Configured session factory.
    """

    def __init__(
        self,
        db_config: DBConfig,
        auto_commit: bool = False,
        expire_on_commit: bool = False,
        auto_flush: bool = False,
    ):
        """
        Initializes the SQLDBClient with the provided database configuration.

        Args:
            db_config: Object containing database connection parameters.
            auto_commit (bool, optional): Whether to enable autocommit. Defaults to False.
            expire_on_commit (bool, optional): Whether to expire objects on commit. Defaults to False.
            auto_flush (bool, optional): Whether to enable autoflush. Defaults to False.
        Raises:
            ValueError: If db_config is missing.
            SQLAlchemyError: If engine creation fails.
        """
        if db_config is None:
            raise ValueError("Missing database configuration.")

        self.auto_commit = auto_commit
        self.auto_flush = auto_flush
        self.expire_on_commit = expire_on_commit

        # Construct database connection string
        self.connection_string = (
            f"{db_config.dialect}://{db_config.username}:{db_config.password}@"
            f"{db_config.host}:{db_config.port}/{db_config.dbname}"
        )
        self.engine = self._create_engine()
        self.session_local = self._create_session()
        print("dbclient initialized")

    def _create_engine(self):
        """
        Creates and returns the SQLAlchemy engine.

        Returns:
            Engine: SQLAlchemy database engine.
        Raises:
            SQLAlchemyError: If engine creation fails.
        """
        try:
            return create_engine(self.connection_string, pool_pre_ping=True)
        except SQLAlchemyError as ex:
            raise SQLAlchemyError(f"Database engine creation failed: {str(ex)}") from ex

    def _create_session(self):
        """
        Creates and returns a session factory.

        Returns:
            sessionmaker: Configured session factory.
        """
        return sessionmaker(
            bind=self.engine,
            autocommit=self.auto_commit,
            autoflush=self.auto_flush,
            expire_on_commit=self.expire_on_commit,
        )

    @contextmanager
    def get_db_session(
        self, auto_commit=False, auto_flush=False, expire_on_commit=False
    ) -> Session:
        """Context manager for obtaining a database session."""

        session = self.session_local()
        session.autocommit = auto_commit
        session.autoflush = auto_flush
        session.expire_on_commit = expire_on_commit

        session = self.session_local()
        try:
            yield session
        except SQLAlchemyError as ex:
            session.rollback()
            raise SQLAlchemyError(f"Session error: {str(ex)}") from ex
        finally:
            session.close()

    def get_session(self) -> Session:
        """Creates and returns a new database session."""
        if self.session_local is None:
            return

        session = self.session_local()
        try:
            yield session
        except SQLAlchemyError as ex:
            session.rollback()
            raise SQLAlchemyError(f"Session error: {str(ex)}") from ex
        finally:
            session.close()
