from CRUD import (get_factures, add_factures, delete_facture_db, get_cust,add_cust, add_prod, get_prod,
                  add_detail_fac,get_detail_fac,add_monitoring,get_all_detail_fact,get_all_produit,
                  get_all_cust,get_all_facture,get_monit)
from model import Facture, Customer
from database import Base, engine, get_db
from fonction_ocr import (webscrapp, OCR, df_ocr, date, test_total_with_errors, rep_tot, tot, concat_dicts,
                        ocr2, prod, prod_final,prep_detail_fac,trouver_multiple)
import logging

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.routing import APIRouter
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi import FastAPI, Depends, HTTPException, Request,status, Cookie, Response, Form, Query

from sqlalchemy.orm import Session

import time
import datetime
from datetime import datetime
import sched
import pandas as pd
import re
import matplotlib.pyplot as plt
import io
import base64
from sqlalchemy.orm import sessionmaker


logging.getLogger('sqlalchemy').setLevel(logging.WARNING)
app = FastAPI()
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

templates = Jinja2Templates(directory="template")
router = APIRouter()

Base.metadata.create_all(bind = engine)


def ocr_boucle(test,k,db: Session):
    dico_qr={}
    dico_texte={}
    count=None
    dico_monit_ocr={}
    for (clé,valeur) in test.items():
        print(valeur)
        dico_monit_ocr[valeur]={}
        try:
            controle= enregistre_facture(db, str(valeur))
        except:
            controle=1
            time.sleep(5)
        if controle==False:
            try:
                dico_texte,dico_qr,message_ocr = ocr2(k,clé,valeur,dico_qr,dico_texte)
                dico_monit_ocr[valeur]['ocr']='OCR bien passer'
            except Exception as e:
                dico_monit_ocr[valeur]['ocr']=e
        else:
            dico_monit_ocr[valeur]['ocr']='deja present bdd'
    return dico_texte,dico_qr,dico_monit_ocr


def enregistre_facture(db: Session, lien):
    test2=get_factures(db, lien)
    if test2:
        a=True
    else:
        a=False
    return a
    

def scrap_week(db: Session,k : int,web : dict):
    test_monit=None
    (resultat_fact,resultat_cust,resultat_prod,resultat_detail_fac) = (None,None,None,None)

    liste_erreur=['message_traitement','message_creation_df_fact','message_crea_df_lie']

    dico_texte, dico_qr,dico_monit_ocr=ocr_boucle(web,k,db)

    for k in dico_monit_ocr.keys():
        for i in liste_erreur:
            dico_monit_ocr[k][i]=None

    if len(dico_texte)==0:
        resultat_fact,resultat_cust,resultat_prod,resultat_detail_fac='deja present bdd','deja present bdd','deja present bdd','deja present bdd'
    else:
        try:
            df_resultat_mid=df_ocr(dico_texte,dico_qr)
            for k in dico_monit_ocr.keys():
                dico_monit_ocr[k]['message_traitement']='traitement du retour OCR bien passé'
        except Exception as e:
            for k in dico_monit_ocr.keys():
                dico_monit_ocr[k]['message_traitement']=e
        else:
            df_resultat_mid.columns.values[-2] = 'fin'
            try:
                erreur = df_resultat_mid.apply(test_total_with_errors, axis=1)
                
                df_erreur = erreur.apply(pd.Series)
                df_erreur.columns = ['resultat', 'erreurs']

                df_resultat_fin=pd.concat([df_resultat_mid,df_erreur],axis=1)

                df=rep_tot(df_resultat_fin)


                df['total']=df.iloc[:,6:].apply(tot,axis=1)

                df['date']=df[1].apply(date)
                df['date']=df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
                df.drop([1],axis=1,inplace=True)
                df['nom']=df[2].apply(lambda x: ' '.join(x[0].split()[2:]) if x is not None and isinstance(x, list) and len(x) > 0 else None)
                df.drop([2],axis=1,inplace=True)
                df['nom_facture']= df[0].apply(lambda x: x[0] if isinstance(x,list) and len(x) > 0 else None)
                df.drop([0],axis=1,inplace=True)
                df['adresse'] = df[4].apply(lambda x: ' '.join(x.split()[1:]) if isinstance(x, str) else None)
                df.drop([4],axis=1,inplace=True)

                df_bdd=df.copy()
                df_bdd['produits'] = df_bdd.loc[:,6:'fin'].apply(concat_dicts, axis=1)

                df_bdd=df_bdd.loc[:,['nom_facture','date','adresse','CUST','nom','CAT','produits','total','erreurs2','resultat']]
                df_bdd['erreurs2'] = df_bdd['erreurs2'].apply(lambda x: str(x) if x else None)
                df_bdd['lien']=df_bdd.index
                df_bdd=df_bdd.reset_index(drop=True)
                for k in dico_monit_ocr.keys():
                    dico_monit_ocr[k]['message_creation_df_fact']='creation df facture bien passé'

            except Exception as e:
                for k in dico_monit_ocr.keys():
                        dico_monit_ocr[k]['message_creation_df_fact']=e
            else:
                try:
                    df_cust=df_bdd.loc[:,['CUST','lien','nom','CAT']]
                    grouped = df_cust.groupby('CUST').agg({'lien': list, 'nom': lambda x: x.iloc[0], 'CAT': lambda x: x.iloc[0]})
                    df_customers = pd.DataFrame(grouped)
                    df_customers=df_customers.reset_index()

                    df_bdd_prod=df_bdd.loc[:,['produits','lien']]

                    df_bdd_prod['produit'] = df_bdd.apply(prod, axis=1)
                    df_produit = df_bdd_prod.groupby('lien')['produit'].apply(lambda x: set().union(*x)).reset_index()
                    df_prod_final=prod_final(df_produit)

                    df_test=df_bdd.loc[:,['lien','produits']]
                    df_test2=prep_detail_fac(df_test)
                        
                    df_bdd['produits'] = df_bdd['produits'].apply(lambda x: str(x))
                    for k in dico_monit_ocr.keys():
                        dico_monit_ocr[k]['message_crea_df_lie']='df bien créer'
 
                except Exception as e:
                    for k in dico_monit_ocr.keys():
                        dico_monit_ocr[k]['message_crea_df_lie']=e

                else:
                    df_bdd=df_bdd.drop(['nom','CAT'],axis=1)

                    resultat_cust=df_customers.apply(add_df_cust, axis=1, args=(db,))
                    resultat_cust=resultat_cust.apply(pd.Series)
                    resultat_cust.columns = ['message_bdd_cust', 'lien']

                    resultat_prod=df_prod_final.apply(add_df_prod, axis=1, args=(db,))
                    resultat_prod=resultat_prod.apply(pd.Series)
                    resultat_prod.columns = ['message_bdd_prod', 'lien']

                    resultat_fact = df_bdd.apply(add_df_facture, axis=1, args=(db,))
                    resultat_fact=resultat_fact.apply(pd.Series)
                    resultat_fact.columns = ['message_bdd_fact', 'lien']

                    resultat_detail_fac = df_test2.apply(add_df_detail_fac, axis=1, args=(db,)) 
                    resultat_detail_fac=resultat_detail_fac.apply(pd.Series)
                    resultat_detail_fac.columns = ['message_bdd_detail_fact', 'lien']
                    
                    merged_df_monit = pd.merge(resultat_fact, resultat_detail_fac, on='lien', how='outer')

                    df_messages_exploded = resultat_prod.explode('lien')

                    merged_df_mid = pd.merge(merged_df_monit, df_messages_exploded, on='lien', how='outer')

                    df_messages_exploded = resultat_cust.explode('lien')

                    df_monit = pd.merge(merged_df_mid, df_messages_exploded, on='lien', how='outer')

                    df_monit_final = pd.DataFrame.from_dict(dico_monit_ocr, orient='index')
                    df_monit_final['lien']=df_monit_final.index
                    df_monit_final=df_monit_final.reset_index(drop=True)
                    df_monit_final = pd.merge(df_monit_final,df_monit, on='lien', how='outer')
                    df_monit_final=df_monit_final.drop_duplicates()
                    print(df_monit_final.shape)

                    test_monit=1
    if test_monit==None:
        liste_colonne=['message_bdd_cust','message_bdd_prod','message_bdd_fact','message_bdd_detail_fact']
        for k in dico_monit_ocr.keys():
            for i in liste_colonne:
                dico_monit_ocr[k][i]=None
        df_monit_final = pd.DataFrame.from_dict(dico_monit_ocr, orient='index')
        df_monit_final['lien']=df_monit_final.index
        df_monit_final=df_monit_final.reset_index(drop=True)
    try :
        df_monit_final.apply(add_df_monit, axis=1, args=(db,))
    except:
        pass
    return df_monit_final

def add_df_facture(ligne, db : Session):
    if ligne['erreurs2'] is not None:
        b=str(ligne['erreurs2'])
    else:
        b=ligne['erreurs2']
    c=str(ligne['produits'])
    try:
        add_factures(db, ligne['nom_facture'],ligne['date'],ligne['adresse'],
                    ligne['CUST'],ligne['total'],b, ligne['resultat'], ligne['lien'])
        a="pas d'erreur"
    except Exception as e:
        a=e
    finally:
        return a,ligne['lien']
    
def add_df_cust(ligne, db : Session):
    a=get_cust(db, ligne['CUST'])
    if not a:
        try:
            add_cust(db,ligne['CUST'],str(ligne['lien']),ligne['nom'],ligne['CAT'])
            b="pas d'erreur"
        except Exception as e:
            b=e
    else:
        b='valeur deja existante'
    return b,ligne['lien']

def add_df_prod(ligne, db : Session):
    a=get_prod(db, ligne['produit'])
    if not a:
        try:
            add_prod(db, ligne['produit'],str(ligne['factures']))
            b="pas d'erreur"
        except Exception as e:
            print(e)
            b=e
    else:
        b='valeur deja existante'
    return b,ligne['factures']

def add_df_detail_fac(ligne, db : Session):
    a=get_detail_fac(db, ligne['lien'],ligne['produit'])
    if not a:
        try:
            add_detail_fac(db, ligne['lien'],ligne['produit'],ligne['quantité'],ligne['prix'],ligne['total'])
            b="pas d'erreur"
        except Exception as e:
            print(e)
            b=e
    else:
        b='valeur deja existante'
    return b,ligne['lien']

def add_df_monit(ligne, db : Session):
    try:
        add_monitoring(db, ligne['ocr'],ligne['message_traitement'],ligne['message_creation_df_fact'],ligne['message_crea_df_lie'],
                       ligne['lien'],ligne['message_bdd_cust'],ligne['message_bdd_prod'],ligne['message_bdd_fact'],ligne['message_bdd_detail_fact'])
    except Exception as e:
        print('!!!!!!!!!!!!!!!!!!!!!!!!erreur monit!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        pass

def validate_date_format(date_str):
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    if date_str:
        if re.match(pattern, date_str):
            return True
        else:
            return False

def trie_fact_date(df,date_debut=None,date_fin=None,col='date'):
    if date_debut :
        try:
            date_debut = pd.to_datetime(str(date_debut))
            df = df[df[col] >= date_debut].sort_values(by=col)
        except:
            pass
    if date_fin:
        try:
            date_fin = pd.to_datetime(str(date_fin))
            df = df[df[col] <= date_fin].sort_values(by=col)   
        except:
            pass 
    return df    

def trie_detail_fact(df1,df2):
    df_filtered = df1[df1['lien'].isin(df2['lien'])]
    return df_filtered


def db_to_df(a):
    df=pd.DataFrame([i.__dict__ for i in a])
    df.drop(columns=['_sa_instance_state'], inplace=True)
    return df

def db_to_series_to_df(a,titre):
    list1 = [i[0] for i in a]
    series = pd.Series(list1)
    df = pd.DataFrame({f"{titre}": series})
    return df

def classement_cust_achat(df):
    df=df.loc[:,['CUST']]
    counts = df['CUST'].value_counts()
    counts=counts.reset_index()
    return counts

import matplotlib.pyplot as plt
import pandas as pd
import io
import base64

def facture_par_periode(df,arm='M',b='de facture'):
    titre=['mois','semaine','jour','année']
    if arm=='M':
        c=0
    if arm=='W':
        c=1
    if arm=='D':
        c=2
    if arm=='Y':
        c=3
    df = df.reset_index()
    weekly_counts = df.resample(arm, on='date').size()
    weekly_counts.index = weekly_counts.index.strftime('%Y-%m-%d')
    
    plt.figure(figsize=(10, 6))
    weekly_counts.plot(kind='bar', color='skyblue')
    
    plt.title(f'Nombre {b} par {titre[c]}')
    plt.xlabel('Date')
    plt.ylabel(f'Nombre {b}')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    html_image = f'<img src="data:image/png;base64,{image_base64}" alt="Graphique">'
    
    with open('temp_graph.png', 'wb') as f:
        f.write(buffer.getvalue())
    
    return html_image

def join_fact_detail(df1,df2,total=True):
    df1=df1.loc[:,['date','lien','CUST']]
    if total==True:
        df2 = df2[df2['produit'].str.contains('TOTAL')]
    elif total==False :
        df2 = df2[~df2['produit'].str.contains('TOTAL')]
    df_final = pd.merge(df1, df2, on='lien', how='inner')
    return df_final

def CUST_depense(df1,df2):
    df=join_fact_detail(df1,df2,total=False)
    aggregated_df = df.loc[:,['CUST','total']].groupby('CUST')['total'].sum().reset_index()
    return aggregated_df

import random

def CA_par_date(df,arm='M'):
    titre=['mois','semaine','jour','année']
    colorr=['yellow','red','green','skyblue','blue']
    if arm=='M':
        c=0
    if arm=='W':
        c=1
    if arm=='D':
        c=2
    if arm=='Y':
        c=3
    b = df.groupby(pd.Grouper(key='date', freq=arm)).sum()
    b.index = b.index.strftime('%Y-%m-%d')
    plt.figure(figsize=(10, 6))
    b['total'].plot(kind='bar', color=colorr[random.randint(0,4)])

    plt.title(f'Argent dépensé par {titre[c]}')
    plt.xlabel('Date')
    plt.ylabel('Argent dépensé (€)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    html_image = f'<img src="data:image/png;base64,{image_base64}" alt="Graphique">'
    
    with open('temp_graph.png', 'wb') as f:
        f.write(buffer.getvalue())
    
    return html_image



@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/add_facture/")

@app.get("/add_facture/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("ajout_fac.html", {"request": request})

@app.post("/add_facture/", response_class=HTMLResponse)
async def connection_user(request: Request,response: Response, demande: bool = Form(None), db: Session = Depends(get_db)):
    if demande is True:
        web,message=webscrapp()
        keys = list(web.keys())
        chunk_size = min(2, 10)
        chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]
        for k, chunk in enumerate(chunks):
            chunk_dict = {key: web[key] for key in chunk}
            resultat = scrap_week(db, k, chunk_dict)

        return templates.TemplateResponse("ajout_fac.html", {"request": request})

@app.get("/path_stat", response_class=HTMLResponse)
async def path_stat_html(request: Request,db: Session = Depends(get_db),date_debut: str = Query(None), date_fin: str = Query(None)):
    a=db_to_df(get_all_detail_fact(db))
    b=db_to_df(get_all_facture(db))
    if date_debut or date_fin:
        b_1=trie_fact_date(b,date_debut,date_fin)
        if len(b_1)>0:
            b=b_1
            a=trie_detail_fact(a,b)
    classement_total_cust=classement_cust_achat(b)
    c1=db_to_df(get_all_cust(db))
    c1=pd.merge(c1,CUST_depense(b,a),on='CUST',how='outer').rename(columns={'total': 'total depense'})
    c=pd.merge(c1,classement_total_cust,on='CUST',how='inner').rename(columns={'count': 'nb de facture'}).to_dict(orient='records')
    d=db_to_series_to_df(get_all_produit(db),'produit').to_dict(orient='records')
    a=a.to_dict(orient='records')
    b=b.to_dict(orient='records')

    return templates.TemplateResponse("path_stat.html", {"request": request,"detail_fact":a,"facture":b,"customer":c,"produit":d})

@app.post("/path_stat2")
async def path_stat_post(date_debut: str = Form(default=None), date_fin : str = Form(default=None)):
    return RedirectResponse(url=f"/path_stat?date_debut={date_debut}&date_fin={date_fin}", status_code=303)

@app.get("/path_graph/", response_class=HTMLResponse)
async def read_root(request: Request,db: Session = Depends(get_db),date:str=Query(None),CUST:str=Query(None),produit:str=Query(None)):
    graph_produit=None
    graph_cust=None
    arm='M'
    if date:
        if date=='mois':
            arm='M'
        if date=='an':
            arm='Y'
        if date=='jour':
            arm='D'
        if date=='semaine':
            arm='W'
    
    a=db_to_df(get_all_detail_fact(db))
    b=db_to_df(get_all_facture(db))
    c=db_to_df(get_all_cust(db))
    d=db_to_series_to_df(get_all_produit(db),'produit').to_dict(orient='records')

    classement_total_cust=classement_cust_achat(b).to_dict(orient='records')
    graph_image=facture_par_periode(b.loc[:,['date','total']],arm)
    if produit:
        df_prod=join_fact_detail(b,a,False)

        df_prod=df_prod.loc[df_prod['produit']==produit]
        print(df_prod)
        graph_produit=facture_par_periode((df_prod.loc[:,['date','CUST']]),arm,f'de produit {produit} vendu')
    if CUST:
        df_cust=b
        df_cust=df_cust.loc[df_cust['CUST']==int(CUST)]
        graph_cust=facture_par_periode(df_cust.loc[:,['date','CUST']],arm,f"d'achat pour {CUST}")

    df_CA=join_fact_detail(b,a)
    graph_image2=CA_par_date(df_CA,arm)

    return templates.TemplateResponse("path_graph.html", {"request": request, 'classement_cust_total':classement_total_cust,
                                                          'graph_image':graph_image,'graph_image2':graph_image2,'graph_produit':graph_produit,
                                                          'graph_cust':graph_cust,'date':date})

@app.post("/path_graph2/", response_class=HTMLResponse)
async def read_root(request: Request,db: Session = Depends(get_db),produit: str=Form(),date : str=Form(None)):
    return RedirectResponse(url=f"/path_graph?produit={produit}&date={date}", status_code=303)

@app.post("/path_graph3/", response_class=HTMLResponse)
async def read_root(request: Request,db: Session = Depends(get_db),CUST: str=Form(),date : str=Form(None)):
    return RedirectResponse(url=f"/path_graph?CUST={CUST}&date={date}", status_code=303)

@app.post("/path_graph4/", response_class=HTMLResponse)
async def read_root(request: Request,db: Session = Depends(get_db),date: str=Form()):
    return RedirectResponse(url=f"/path_graph?date={date}", status_code=303)

@app.get("/path_stat/CUST", response_class=HTMLResponse)
async def read_root(request: Request,CUST : str = Query(),db: Session = Depends(get_db)):
    a=db_to_df(get_all_detail_fact(db))
    b=db_to_df(get_all_facture(db))
    df=join_fact_detail(b,a,total=0)
    print(df.dtypes)
    df1=df.loc[df['CUST'] == int(CUST)].reset_index(drop=True).to_dict(orient='records')
    df2=b.loc[b['CUST'] == int(CUST)].reset_index(drop=True).to_dict(orient='records')
    return templates.TemplateResponse("stat_cust.html",{"request": request, "detail_facture":df1, "facture" : df2 })

@app.get("/path_monit/", response_class=HTMLResponse)
async def read_monit(request: Request,db: Session = Depends(get_db),date_debut: str = Query(None), date_fin: str = Query(None)):
    df=db_to_df(get_monit(db))
    if date_debut or date_fin:
        df=trie_fact_date(df,date_debut,date_fin,col='timestamp')
    monit_ocr= df.loc[~df['ocr'].str.contains('OCR bien passer')].to_dict(orient='records')
    df['message_traitement'] = df['message_traitement'].fillna('')
    df['message_creation_df_fact'] = df['message_creation_df_fact'].fillna('')
    df['message_crea_df_lie'] = df['message_crea_df_lie'].fillna('')
    df['message_bdd_cust'] = df['message_bdd_cust'].fillna('')
    df['message_bdd_prod'] = df['message_bdd_prod'].fillna('')
    df['message_bdd_fact'] = df['message_bdd_fact'].fillna('')
    df['message_bdd_detail_fact'] = df['message_bdd_detail_fact'].fillna('')
    monit_trait = df.loc[~df['message_traitement'].str.contains('traitement du retour OCR bien passé')].to_dict(orient='records')
    monit_crea_df1 = df.loc[~df['message_creation_df_fact'].str.contains('creation df facture bien passé')].to_dict(orient='records')
    monit_crea_df2 = df.loc[~df['message_crea_df_lie'].str.contains('df bien créer')].to_dict(orient='records')
    monit_bdd_cust = df.loc[~df['message_bdd_cust'].str.contains("pas d'erreur")].to_dict(orient='records')
    monit_bdd_prod = df.loc[~df['message_bdd_prod'].str.contains("pas d'erreur")].to_dict(orient='records')
    monit_bdd_fact = df.loc[~df['message_bdd_fact'].str.contains("pas d'erreur")].to_dict(orient='records')
    monit_bdd_detail_fact = df.loc[~df['message_bdd_detail_fact'].str.contains("pas d'erreur")].to_dict(orient='records')
    df=df.to_dict(orient='records')
    return templates.TemplateResponse("path_monit.html",{"request": request, "monit":df,"monit_ocr":monit_ocr, "monit_trait":monit_trait,
                                                         "monit_crea_df1":monit_crea_df1,"monit_crea_df2":monit_crea_df2,
                                                         "monit_bdd_cust":monit_bdd_cust,"monit_bdd_prod":monit_bdd_prod,
                                                         "monit_bdd_fact":monit_bdd_fact,"monit_bdd_detail_fact":monit_bdd_detail_fact})

@app.post("/path_monit2")
async def path_monit_post(date_debut: str = Form(default=None), date_fin : str = Form(default=None)):
    return RedirectResponse(url=f"/path_monit?date_debut={date_debut}&date_fin={date_fin}", status_code=303)