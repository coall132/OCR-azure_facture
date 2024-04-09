from sqlalchemy import create_engine, MetaData, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
import urllib
import pyodbc
import logging

logging.getLogger('sqlalchemy').setLevel(logging.WARNING)

load_dotenv()

BDD_URL = os.environ['url_bdd2']

quoted_params = urllib.parse.quote_plus(BDD_URL)

BDD_URL


conn_str = 'mssql+pyodbc:///?odbc_connect={}'.format(quoted_params)

try:
    engine = create_engine(conn_str, echo=True)

except Exception as e:
    print('Error')

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

