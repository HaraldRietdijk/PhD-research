from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import scoped_session, sessionmaker
from database.models.hft_functions import make_functions
from database.models.model import DBActivityModel

__all__ = ["Query", "create_engine_and_session"]

def create_engine_and_session(app, schema):
    engine_url='mysql+pymysql://vfc:vfc@localhost:3306/'+ schema
    print('Setting up engine tot connect to:')
    print(engine_url)

    app.engine = create_engine(engine_url, pool_pre_ping=True)
    app.session = scoped_session(
        sessionmaker(
            autocommit=False,
            autoflush=True,
            binds={DBActivityModel: app.engine},
        )
    )

def setup_db(app, force, testing, schema):
    table_to_check="hft_data_t"    
    if testing:
        table_to_check="stuff"
    if not(inspect(app.engine).has_table(table_to_check, schema=schema)) or force:  
        if not testing:
            make_functions()
        DBActivityModel.metadata.drop_all(bind=app.engine)
    DBActivityModel.metadata.create_all(bind=app.engine)
