from sqlalchemy.orm import Session
from model import Facture, Customer, Produit, Detail_fact, Monitoring
from schema import FactureCreate, CustomerCreate
from database import get_db, engine
import pandas as pd
from datetime import datetime
from sqlalchemy import and_

def get_factures(db: Session, lien : str):
    return db.query(Facture).filter(Facture.lien == lien).first()


def get_cust(db: Session, CUST : int):
    return db.query(Customer).filter(Customer.CUST == CUST).first()

def get_prod(db: Session, produit : int):
    return db.query(Produit).filter(Produit.produit == produit).first()

def get_detail_fac(db: Session, lien: str, produit: str):
    return db.query(Detail_fact).filter(and_(Detail_fact.lien == lien, Detail_fact.produit == produit)).first()

def get_monit(db: Session):
    return db.query(Monitoring).all()

def add_factures(db : Session, nom_facture : str, date : datetime, adresse : str, CUST : int,
                 total : int, erreurs2 : str, resultat : str, lien : str):
    
    db_facture= Facture(nom_facture=nom_facture,date=date,adresse=adresse,CUST=CUST,
                        total=total,erreurs2=erreurs2,resultat=resultat, lien= lien)
    db.add(db_facture)
    db.commit()
    db.refresh(db_facture)
    return db_facture

def add_cust(db : Session, CUST :int, factures_passe : str,nom : str, CAT : str):
    db_customer= Customer(CUST=CUST,nom=nom,CAT=CAT)
    db.add(db_customer)
    db.commit()
    db.refresh(db_customer)
    return db_customer

def add_prod(db : Session, produit : str, factures_passe : str):
    db_prod= Produit(produit=produit)
    db.add(db_prod)
    db.commit()
    db.refresh(db_prod)
    return db_prod

def add_detail_fac( db : Session, lien : str, produit : str, quantite : int, prix : float, total : float):
    db_detail_fac= Detail_fact(lien=lien,produit=produit,quantite=quantite,prix=prix,total=total)
    db.add(db_detail_fac)
    db.commit()
    db.refresh(db_detail_fac)
    return db_detail_fac

def add_monitoring(db : Session, ocr : str, message_traitement : str, message_creation_df_fact : str, message_crea_df_lie : str,
                    lien : str, message_bdd_cust : str, message_bdd_prod : str, message_bdd_fact : str, message_bdd_detail_fact : str):
    
    db_monitoring= Monitoring(ocr=ocr,message_traitement=message_traitement, message_creation_df_fact= message_creation_df_fact,
                                message_crea_df_lie= message_crea_df_lie,lien=lien,message_bdd_cust=message_bdd_cust,message_bdd_prod=message_bdd_prod,
                                message_bdd_fact=message_bdd_fact,message_bdd_detail_fact=message_bdd_detail_fact)
    db.add(db_monitoring)
    db.commit()
    db.refresh(db_monitoring)
    return db_monitoring

def delete_facture_db(db: Session, nom_facture : str):
    facture = get_factures(db, nom_facture)
    if not facture:
        raise ValueError('Wrong username')
    db.delete(facture)
    db.commit()

def get_facture_date_lien(db: Session, date_debut: datetime,date_fin):
    return db.query(Facture.lien).filter(and_(Facture.date > date_debut,Facture.date<date_fin)).all()

def get_facture_date(db: Session, date_debut: datetime,date_fin):
    return db.query(Facture).filter(and_(Facture.date > date_debut,Facture.date<date_fin)).all()

def get_all_produit(db:Session):
    return db.query(Produit.produit).all()

def get_all_cust(db:Session):
    return db.query(Customer).all()

def get_all_detail_fact_lien(db:Session, lien):
    return db.query(Detail_fact).filter(Detail_fact.lien == lien).all()

def get_all_facture(db:Session):
    return db.query(Facture).all()

def get_all_detail_fact(db:Session):
    return db.query(Detail_fact).all()




