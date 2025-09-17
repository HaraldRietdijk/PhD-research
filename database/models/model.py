from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base

__all__ = ["DBActivityModel"]

DBActivityModel = declarative_base(
    metadata=MetaData(
        naming_convention=dict(
            pk="pk_%(table_name)s",
            fk="fk_%(table_name)s_%(column_0_name)s",
            ix="ix_%(column_0_label)s",
            uq="uq_%(table_name)s_%(column_0_name)s",
            ck="ck_%(table_name)s_%(column_0_name)s",
        )
    )
)