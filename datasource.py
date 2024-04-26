from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, MetaData, Table, select

def get_db_connection():
    CONNECTION_STRING="postgresql+psycopg2://myuser:mypassword@localhost:5428/mydatabase2"

    # Assuming `engine` is your SQLAlchemy engine
    new_engine = create_engine(url=CONNECTION_STRING)
    NewSession = sessionmaker(autoflush=False, bind=new_engine)

    # Reflect the metadata
    metadata = MetaData()
    langchain_storage_collection = Table('langchain_storage_collection', metadata, autoload_with=new_engine)
    langchain_storage_items = Table('langchain_storage_items', metadata, autoload_with=new_engine)

    with NewSession() as session:
        # Perform a query
        collection_uuid = str((session.execute(select(langchain_storage_collection.c.uuid).where(langchain_storage_collection.c.name == COLLECTION_NAME)).fetchone())[0])
        
        docs_uuid = session.execute(select(langchain_storage_items.c.uuid).where(langchain_storage_items.c.collection_id == collection_uuid)).fetchall()
        combined_id_doc_keys = [str(result[0]) for result in docs_uuid]