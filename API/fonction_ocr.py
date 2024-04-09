import os
import requests 
import shutil
from bs4 import BeautifulSoup
from mimetypes import guess_extension
import re
import random
import tempfile
import os
from dotenv import load_dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import os
from dotenv import load_dotenv
from array import array
from PIL import Image
import sys
import time
import base64
import pandas as pd

import aspose.barcode as barcode

import pyzbar.pyzbar
import PIL.Image

import re
from datetime import datetime


load_dotenv()

def webscrapp():
    image_dico = {}
    count=None
    while True:
        main_url = 'https://invoiceocrp3.azurewebsites.net/invoices?start_date='

        if len(image_dico) > 0:
            last_key = list(image_dico.keys())[-1]
            last_key =last_key.split('__')[0]
            main_url += last_key
        else:
            pass
        try:
            response = requests.get(main_url)
            message='reponse API facture ok'
        except Exception as e:
            message=e
        else:
            soup = BeautifulSoup(response.text, 'html.parser')

            base_url = 'https://invoiceocrp3.azurewebsites.net'
            li_balises = soup.find_all('li')

            for li_balise in li_balises:
                text = '__'.join(li_balise.text.strip().split(" ")[1:])
                href = li_balise.find('a').get('href') 
                image_dico[text] = base_url + href

            if count == list(image_dico.keys())[-1]:
                break
            count=list(image_dico.keys())[-1]

    return image_dico,message

def read_qr_code(filename):
    image = PIL.Image.open(filename)
    codes = pyzbar.pyzbar.decode(image)
    return codes[0].data.decode() if codes else None

def enhance_contrast_and_sharpness(image_url,contrast=3.5):
    try:
        response = requests.get(image_url)
    except:
        return None
    image = Image.open(BytesIO(response.content))
    
    if image.mode == "RGBA":
        image = image.convert("RGB")
    
    contrast_enhancer = ImageEnhance.Contrast(image)
    enhanced_image_contrast = contrast_enhancer.enhance(contrast)

    sharpened_image = enhanced_image_contrast.filter(ImageFilter.SHARPEN)


    image_final = sharpened_image.convert("L")
    bytes_image = BytesIO()
    
    image_final.save(bytes_image, format='JPEG')
    bytes_image.seek(0)
    
    return bytes_image.getvalue()

def OCR(test,contrast=3.5):
    dico_qr={}
    dico_texte={}
    try:
        endpoint = os.environ["OCR_ENDPOINT"]
        key = os.environ["OCR_KEY"]
    except KeyError:
        print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
        print("Set them before running this sample.")
        exit()
    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))
    for k,(clé,valeur) in enumerate(test.items()):
        if k==150:
            try:
                computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))
            except:
                pass
            
        dico_texte[valeur]=[]
        
        enhanced_image = enhance_contrast_and_sharpness(valeur,contrast)
        if enhanced_image==None:
            pass
        else:
            with tempfile.NamedTemporaryFile(delete=False) as temp_image:
                temp_image.write(enhanced_image)
                temp_image_path = temp_image.name
    
            with open(temp_image_path, "rb") as image_file:
                try:
                    read_response = computervision_client.read_in_stream(image_file, raw=True)
                except:
                    print('erreur Azure')
                else:  
                    read_operation_location = read_response.headers["Operation-Location"]
                    operation_id = read_operation_location.split("/")[-1]
            
            
                    while True:
                        read_result = computervision_client.get_read_result(operation_id)
                        if read_result.status not in ['notStarted', 'running']:
                            break
                        time.sleep(1)
                    print(f"Image {k}")
                    if read_result.status == OperationStatusCodes.succeeded:
                        for text_result in read_result.analyze_result.read_results:
                            for line in text_result.lines:
                                dico_texte[valeur].append({line.text:line.bounding_box})
                                #print(f"texte={line.text}, axe={line.bounding_box}")
                    url = read_qr_code(temp_image_path)
                    url2=url.split("\n")
                    for k,i in enumerate(url2):
                        url2[k]=i.split(":",1)
                    url2=dict(url2)
                    del url2['DATE']
                    del url2['INVOICE']
                    dico_qr[valeur]=url2
                                
            os.unlink(temp_image_path)
    return dico_texte, dico_qr

def ocr2(k,clé,valeur,dico_qr,dico_texte,computervision_client=None,contrast=3.5):
    try:
        endpoint = os.environ["OCR_ENDPOINT"]
        key = os.environ["OCR_KEY"]
    except KeyError:
        print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
        print("Set them before running this sample.")
        exit()

    try:
        computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))
    except:
        message='erreur API OCR (connection ComputerVisionClient)'
    else:
        dico_texte[valeur]=[]
        
        enhanced_image = enhance_contrast_and_sharpness(valeur,contrast)
        if enhanced_image==None:
            pass
        else:
            with tempfile.NamedTemporaryFile(delete=False) as temp_image:
                temp_image.write(enhanced_image)
                temp_image_path = temp_image.name

            with open(temp_image_path, "rb") as image_file:
                try:
                    read_response = computervision_client.read_in_stream(image_file, raw=True)
                except:
                    message='erreur API OCR (erreur reponse)'
                else:  
                    read_operation_location = read_response.headers["Operation-Location"]
                    operation_id = read_operation_location.split("/")[-1]
            
            
                    while True:
                        read_result = computervision_client.get_read_result(operation_id)
                        if read_result.status not in ['notStarted', 'running']:
                            break
                        time.sleep(1)
                    if read_result.status == OperationStatusCodes.succeeded:
                        for text_result in read_result.analyze_result.read_results:
                            for line in text_result.lines:
                                dico_texte[valeur].append({line.text:line.bounding_box})
                    url = read_qr_code(temp_image_path)
                    url2=url.split("\n")
                    for k,i in enumerate(url2):
                        url2[k]=i.split(":",1)
                    url2=dict(url2)
                    del url2['DATE']
                    del url2['INVOICE']
                    dico_qr[valeur]=url2
                    message='OCR bien passer'
                                    
            os.unlink(temp_image_path)
    return dico_texte, dico_qr,message        
    
    
def ocr_boucle(test):
    dico_qr={}
    dico_texte={}
    for k,(clé,valeur) in enumerate(test.items()):
        print(valeur)
        if k==0:
            dico_texte,dico_qr,computervision_client = ocr2(k,clé,valeur,dico_qr,dico_texte)
        else:
            dico_texte,dico_qr,computervision_client = ocr2(k,clé,valeur,dico_qr,dico_texte,computervision_client)
    return dico_texte,dico_qr
    
def ligne(a):
    b=[]
    c=[]
    for (key,value) in a.items():
        if value is not None:
            pos=int(list(value.values())[0][7])
            if (pos) not in b:
                c.append(list(value.keys()))
            else:
                c[-1]+=list(value.keys())
            b=[pos+(i-10) for i in range(0,21)]

    return c

def ligne2(a):
    c=[]
    deja_mis=[]
    for (key,value) in a.items():
        if value is not None:
            if list(value.values())[0] not in deja_mis:
                pos=int(list(value.values())[0][7])
                b=[pos+(i-7) for i in range(0,15)]
                c.append(list(value.keys()))
                deja_mis.append(list(value.values())[0])
                for (key2,value2) in a.items():
                    if value2 is not None:
                        if value2!=value:
                            if list(value2.values())[0] not in deja_mis:
                                pos2=int(list(value2.values())[0][7])
                                if pos2 in b:
                                    c[-1]+=list(value2.keys())
                                    deja_mis.append(list(value2.values())[0])
    return c

def colle_calcul(a):
    b=[]
    nb=''
    phrase=''
    error=['XX', ' X ']
    error2='X'
    if a is not None:
        for el in a:
            if len(el)>0:
                while True:
                    if not el:
                        break
                    if len(el)==0:
                        break
                    if not el[0].isalnum():
                        el=el[1:]
                    else:
                        break
                if el:
                    if len(el)>0:
                        if el[0].isdigit():
                            nb+=' '+el
                        elif error[0] not in el and error[1] not in el and el!=error2:
                            phrase+=' '+el
        b.append(phrase)
        b.append(nb)
    else:
        b=None
    return b 

def class_article(a):
    if a is not None:
        if len(a)==2:
            a=supp_esp_point(a)
            dico={}
            dico[a[0]]=[]
            prix=r'(\d+\.\d+)'
            nb=r'(\d+)'
            b=re.search(prix,a[1])
            c=re.search(nb,a[1])
            if b:
                part1=b.group(0)
                part2=c.group(0)
                test=part1.split('.')
                if test[0]==part2:
                    verification=r'(?<!\.)\b(\d+)\b(?!\.)'
                    d=re.search(verification,a[1])
                    if d:
                        dico[a[0]].append(d.group(0))
                        dico[a[0]].append(part1)
                        return dico
                    else:
                        dico[a[0]].append('?')
                        dico[a[0]].append(part1)
                        return dico
                else:
                    dico[a[0]].append(part2)
                    dico[a[0]].append(part1)
                    return dico
    return a

def supp_esp_point(a):
    stri=''
    for k,i in enumerate(a[1]):
        if k!=0:
            if i==' ' and a[1][k-1]=='.':
                pass
            else:
                stri+=i
        else:
            stri+=i
    a[1]=stri
    return a
                

def df_ocr(dico_texte, dico_qr):
    for valeur in dico_texte.values():
        for i in valeur:
            if any(len(cle) == 1 and (not cle.isalnum() or cle == 'x') for cle in i.keys()):
                valeur.remove(i)  
    qr = pd.DataFrame.from_dict(dico_qr, orient='index')
    texte= pd.DataFrame.from_dict(dico_texte, orient='index')
    resultat = texte.apply(ligne2, axis=1)
    df_resultat = pd.DataFrame(resultat.tolist(), index=resultat.index)
    df_resultat.iloc[:, 6:] = df_resultat.iloc[:, 6:].applymap(colle_calcul)
    df_resultat=df_resultat.drop(3,axis=1)
    df_resultat[4]=df_resultat[4]+df_resultat[5]

    df_resultat=pd.concat([qr,df_resultat],axis=1)
    df_resultat=df_resultat.drop(5,axis=1)
    df_resultat[4] = df_resultat[4].apply(lambda x: ' '.join(x))
    df_resultat_2=df_resultat.copy()
    df_resultat_2.iloc[:, 6:] = df_resultat_2.iloc[:, 6:].applymap(class_article)
    return df_resultat_2

def test_total_with_errors(ligne):
    total = 0
    dico_erreur = {}

    for col, valeur in ligne.items():
        if valeur is not None and isinstance(valeur, dict):
            if list(valeur.keys())[0] != ' TOTAL':
                if list(valeur.values())[0][0] != '?':
                    total += int(list(valeur.values())[0][0]) * float(list(valeur.values())[0][1])
                elif list(valeur.values())[0][0] == '?':
                    dico_erreur[list(valeur.keys())[0]] = float(list(valeur.values())[0][1])
                        
            elif list(valeur.keys())[0] == ' TOTAL':
                if total - round(float(list(valeur.values())[0][1]), 2) < 1 and total - round(float(list(valeur.values())[0][1]), 2) > -1:
                    return True, None
                else:
                    if len(dico_erreur)==0:
                        return [round(float(list(valeur.values())[0][1]), 2),round(total, 2)], None
                    else:
                        return [round(float(list(valeur.values())[0][1]), 2),round(total, 2)], dico_erreur
   
    return None, None
                


def reparation_ocr(a,b):
    dif=b[0]-b[1]
    res_int=dif//a
    res_float=dif/a
    ecart = abs(res_float - res_int)
    if ecart<0.05:
        return res_int
    elif ecart>0.95:
        return (res_int+1)
    else:
        return '?'
    
def rep_tot(df):
    valeurs_differentes = df['resultat'] != True
    if valeurs_differentes.any():
        for index, ligne in df.iterrows():
            if ligne['resultat'] != True:
                a = ligne['erreurs']
                b = ligne['resultat']
                if a is not None:
                    for key, valeur in ligne.items():
                        if isinstance(valeur, dict):
                            for clé in a.keys():
                                if list(valeur.keys())[0] == clé:
                                    if isinstance(list(valeur.values())[0], list):
                                        if list(valeur.values())[0][0] == '?' and len(a)==1:
                                            g = reparation_ocr(list(a.values())[0], b)
                                            df.at[index, key][clé][0]=str(int(g))
        erreur = df.apply(test_total_with_errors, axis=1)
        df_erreur = erreur.apply(pd.Series)
        df_erreur.columns = ['resultat', 'erreurs2']
        
        df=df.drop(['resultat'],axis=1)
        df_resultat_fin=pd.concat([df,df_erreur],axis=1)
    else:
        df = df.rename(columns={'erreurs': 'erreurs2'})
        df_resultat_fin=df
        
    return df_resultat_fin

def tot(a):
    b=None
    if isinstance(a,dict):
        for key,value in a.items():
            if isinstance(value,dict):
                for clé,valeur in value.items():
                    if clé==' TOTAL':
                        b=valeur[1]
                        b=float(b)
    return b

def date(a):
    b=a[0].split()
    d=[]
    for i in b:
        try :
            int(i[0])
        except:
            pass
        else:
            d.append(i)
    c=' '.join(d)
    return c


def tris_df_date_intervalle(df,date_debut,date_fin):
    masque = (df['date'] >= date_debut) & (df['date'] <= date_fin)
    nouveau_df = df[masque]
    return nouveau_df

def tris_df_date(df,month=None,year=None,week=None,day=None):
    if year:
        df = df[df['date'].dt.year == year]
    if month:
        df = df[df['date'].dt.month == month]
    if day:
        df = df[df['date'].dt.day == day] 
    if week:
       df= df[df['date'].dt.isocalendar().week == week]
    return df

def prod(ligne):
    produits = set()
    for produit in ligne['produits']:
        produits.update(produit.keys())
    return produits

def prod_final(df):
    produits_par_facture = {}
    for index, row in df.iterrows():
        for produit in row['produit']:
            
            if produit not in produits_par_facture:
                produits_par_facture[produit] = []
            produits_par_facture[produit].append(row['lien'])

    df_final = pd.DataFrame(produits_par_facture.items(), columns=['produit', 'factures'])
    return df_final

def concat_dicts(row):
    dicts = [val for val in row if isinstance(val, dict)]
    return dicts

def prep_detail_fac(df_test):
    df_test = df_test.explode('produits')
    
    df_test['produit']=df_test['produits'].apply(lambda x: list(x.keys())[0])
    df_test['quantité']=df_test['produits'].apply(lambda x: list(x.values())[0][0])
    df_test['quantité']=df_test['quantité'].replace('?',1)
    df_test['prix']=df_test['produits'].apply(lambda x: list(x.values())[0][1])
    df_test['total'] = df_test.apply(lambda row: round(int(row['quantité']) * float(row['prix']), 2), axis=1)
    df_test['quantité']=df_test['quantité'].apply(lambda x: int(x))
    df_test['prix']=df_test['prix'].apply(lambda x: float(x))
    
    df_test=df_test.drop('produits',axis=1)
    return df_test

def trouver_multiple(a, b):
    multiple = b + (a - (b % a))
    if multiple % a == 1:
        return multiple + a
    return multiple



