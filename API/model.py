from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, Float, ForeignKeyConstraint, func, PrimaryKeyConstraint, select
from sqlalchemy.orm import relationship
from database import Base, engine
from sqlalchemy_utils import create_materialized_view,refresh_materialized_view


class Facture(Base):
    __tablename__ = "factures"
    __table_args__ = {'schema':'dbo'}

    id = Column(Integer,autoincrement=True, primary_key=True)
    lien = Column(String(255), unique=True)
    nom_facture = Column(String)
    date = Column(DateTime)
    adresse = Column(String)
    CUST = Column(Integer, ForeignKey('dbo.customers.CUST'))
    total = Column(Integer)
    erreurs2 = Column(String)
    resultat = Column(String)


class Customer(Base):
    __tablename__ = "customers"
    __table_args__ = {'schema':'dbo'}  

    CUST = Column(Integer, primary_key=True)
    nom = Column(String)
    CAT = Column(String)


class Detail_fact(Base):
    __tablename__ = "details_facture"
    __table_args__ = {'schema':'dbo'} 

    lien = Column(String(255), ForeignKey('dbo.factures.lien'), primary_key=True)
    produit = Column(String(255), ForeignKey('dbo.products.produit'), primary_key=True)
    quantite = Column(Integer,primary_key=True)
    prix = Column(Float)
    total = Column(Float)


class Produit(Base):
    __tablename__ = "products"
    __table_args__ = {'schema':'dbo'}  

    id = Column(Integer,autoincrement=True, primary_key=True)
    produit = Column(String(255), unique=True)


Customer.__table_args__ = (
    ForeignKeyConstraint(['CUST'], ['factures.CUST'], name='fk_customers_factures_cust'),
    {'schema':'dbo'}
)

Detail_fact.__table_args__ = (
    ForeignKeyConstraint(['lien'], ['factures.lien'], name='fk_details_facture_factures_lien'),
    {'schema':'dbo'}
)

Produit.__table_args__ = (
    ForeignKeyConstraint(['produit'], ['details_facture.produit'], name='fk_products_details_facture_produit'),
    {'schema':'dbo'}
)

class Monitoring(Base):
    __tablename__ = "monitoring"
    __table_args__ = {'schema':'dbo'} 

    id = Column(Integer,autoincrement=True, primary_key=True)
    ocr = Column(String)
    message_traitement = Column(String)
    message_creation_df_fact = Column(String)
    message_crea_df_lie = Column(String)
    lien = Column(String)
    message_bdd_cust = Column(String)
    message_bdd_prod = Column(String)
    message_bdd_fact = Column(String)
    message_bdd_detail_fact = Column(String)
    timestamp = Column(DateTime, default=func.now())


Base.metadata.create_all(bind=engine)
