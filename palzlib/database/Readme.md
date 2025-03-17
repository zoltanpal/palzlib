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

```
db_config.py
```
Provides the DBConfig class for handling database configuration parameters.

**Features**:
- Uses Python's dataclass for structured configuration.
- Includes validation for required fields.
- Provides default values for dialect and port.

**Usage:**
```pyhton
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
