from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class FactureBase(BaseModel):
    nom_facture : Optional[str]
    date : Optional[datetime]
    adresse : Optional[str]
    CUST : Optional[int]
    produits : Optional[str]
    total : Optional[int]
    erreurs2 : Optional[str]
    resultat : Optional[str]
    lien : Optional[str]
class FactureCreate(FactureBase):
    pass
class Facture(FactureBase):
    id : int
    class Config:
        from_attributes = True

class CustomerBase(BaseModel):
    CUST : int
    nom : str
    CAT : str
class CustomerCreate(CustomerBase):
    pass
class Customer(CustomerBase):
    pass
    class Config:
        from_attributes = True

class ProductBase(BaseModel):
    produit : str
class ProductCreate(ProductBase):
    pass
class Product(ProductBase):
    id : int
    class Config:
        from_attributes = True

class Detail_factBase(BaseModel):
    lien : str
    produit : str
    quantite : Optional[int]
    prix : Optional[float]
    total : Optional[float]
class ProductCreate(Detail_factBase):
    pass
class Product(Detail_factBase):
    pass
    class Config:
        from_attributes = True

class MonitoringBase(BaseModel):
    ocr : Optional[str]
    message_traitement : Optional[str]
    message_creation_df_fact : Optional[str]
    message_crea_df_lie : Optional[str]
    lien : Optional[str]
    message_bdd_cust : Optional[str]
    message_bdd_prod : Optional[str]
    message_bdd_fact : Optional[str]
    message_bdd_detail_fact : Optional[str]
class ProductCreate(MonitoringBase):
    pass
class Product(MonitoringBase):
    id : int
    class Config:
        from_attributes = True