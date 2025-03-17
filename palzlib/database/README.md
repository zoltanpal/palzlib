# SQLAlchemy database module

This module provides a database client and mapping functionality using SQLAlchemy.
It enables dynamic table mapping and session management.

## Requirements
- Python 3.10+
- SQLAlchemy

## Notes
Ensure the database is accessible and configured correctly before using this module.
For error handling, exceptions raised during session operations should be logged appropriately.

## Files

### db_config.py

Provides the DBConfig class for handling database configuration parameters.

**Features**:
- Uses Python's dataclass for structured configuration.
- Includes validation for required fields.
- Provides default values for dialect and port.

**Usage:**
```python
from palzlib.database.db_config import DBConfig

# Initialize configuration
config = DBConfig(
    username="user",
    password="password",
    dbname="mydatabase",
    host="localhost"
)

print(config)
```

### db_client.py
Provides the DBClient class for managing database connections and sessions.

**Features:**
- Establishes database connections using SQLAlchemy.
- Supports autocommit, autoflush, and expiration settings.
- Provides a context manager for handling database sessions safely.

**Usage:**
```python
from palzlib.database.db_client import DBClient
from palzlib.database.db_mapper import DBMapper
from palzlib.database.db_config import DBConfig

# Initialize configuration
config = DBConfig(
    dialect="postgresql",
    username="user",
    password="password",
    host="localhost",
    port=5432,
    dbname="mydatabase"
)

# Create a database client
db_client = DBClient(config)

# Get a session
with db_client.get_db_session() as session:
    result = session.execute("SELECT * FROM users")
    print(result.fetchall())
```

### db_mapper.py
Provides the DBMapping class for dynamically mapping database tables to Python classes.

**Features:**
- Uses SQLAlchemy’s automap feature to reflect and map tables.
- Enables accessing mapped tables via attributes, dictionary keys, or a lookup method.
- Customizes relationships between tables.

**Usage:**
```python
from palzlib.database.db_mapper import DBMapper

# Initialize mapping
db_mapper = DBMapper(db_client)

# Access a mapped table
User = db_mapper.get_model("users")
# or
User = db_mapper.db_classes.users

# Query using the mapped class
with db_client.get_db_session() as session:
    users = session.query(User).all()
    print(users)
```
