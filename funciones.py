
from googlesearch import search     #realiza busquedas en google
from serpapi import GoogleSearch    #busqueda inversa de imagenes
from bs4 import BeautifulSoup       #scrapy de resuestas http
import requests, html5lib           #peticiones http                 
import re, string, unicodedata      #manejo de texto
import numpy as np                  
from textblob import TextBlob       #correccion de texto
from autocorrect import Speller     #correccion de texto     
from difflib import SequenceMatcher #similitud de letras en textos
from string import punctuation      #manejo de texto
from sklearn.metrics.pairwise import cosine_similarity        #coseno para similitud de textos
from sklearn.feature_extraction.text import TfidfVectorizer   #similitud de textos
import json                         #manejo de archivos json
import pandas as pd                 #manejo de archivos 
from gensim.corpora import Dictionary       #diccionario de palabras
from gensim.models import LdaModel          #modelo mdl para topicos
import tensorflow as tf                     #red neuronal
import random                               #generacion aleatoria
import matplotlib.pyplot as plt             #dibujo de graficas
from wordcloud import WordCloud       
import nltk
from nltk.corpus import stopwords           #preprocesamiento stopwords
from nltk.stem import SnowballStemmer       #preprocesamiento stemming
from nltk.tokenize import ToktokTokenizer   #preprocesamiento tokenizer
nltk.download('stopwords')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

spell = Speller(lang='es')
vectorizer = TfidfVectorizer ()
language_stopwords = stopwords.words('spanish')
non_words = list(punctuation)
tokenizer = ToktokTokenizer()
STOPWORDS = set(stopwords.words("spanish"))
stemmer = SnowballStemmer("spanish")
analyzer = SentimentIntensityAnalyzer()
session = requests.Session()
headers = {'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.106 Safari/537.36 OPR/38.0.2220.41'}
#busqueda en google

def busqueda_consulta(texto):
  tld = "com" 
  lang = "es" 
  num=10 
  start=0 
  stop=num 
  pause=2.0 
  results = search(texto, tld=tld, lang=lang, num=num, start=start, stop=stop, pause=pause)
  return results
#consulta a una url
def consulta_url(texto):
  return requests.get(texto,headers=headers)

def consulta_titulo(texto):
  query = texto
  search = query.replace(' ', '+')
  results = 10
  url = (f"https://www.google.com/search?q={search}&num={results}")

  requests_results = requests.get(url)
  soup_link = BeautifulSoup(requests_results.content, "html.parser")
  links = soup_link.find_all("a")
  v = []
  for link in links:
    link_href = link.get('href')
    #print(link_href)
    if "url?q=" in link_href and not "webcache" in link_href:
      title = link.find_all('h3')
      if len(title) > 0:
          v.append(link.get('href').split("?q=")[1].split("&sa=U")[0])
          #print(title[0].getText())
  return v

def separar_mayusculas(texto):
  aux = ""
  for i in range(len(texto)):
    if not texto[i].isdigit() and texto[i] == texto[i].upper():
      aux = aux + " " + texto[i]
    else:      
      aux = aux + texto[i]
  return aux

def separar_numeros(texto):
  aux = ""
  for i in range(len(texto)):
    if texto[i].isdigit() and not texto[i-1].isdigit():
      aux = aux +" "+ texto[i]
    else:
      aux = aux + texto[i]
  return aux

def remplazar_ansi(cad):
  cad = cad.replace("\\u00c1","Á")
  cad = cad.replace("\\u00e1","á")
  cad = cad.replace("\\u00c9",'É')
  cad = cad.replace("\\u00e9",'é')
  cad = cad.replace("\\u00cd",'Í')
  cad = cad.replace("\\u00ed",'í')
  cad = cad.replace("\\u00d3",'Ó')
  cad = cad.replace("\\u00f3",'ó')
  cad = cad.replace("\\u00da",'Ú')
  cad = cad.replace("\\u00fa",'ú')
  cad = cad.replace("\\u00d1",'Ñ')
  cad = cad.replace("\\u00f1",'ñ')
  cad = cad.replace("\\n"," ")
  return cad

def limpieza_html(texto):
  texto = BeautifulSoup(str(texto), "lxml").text
  texto = separar_mayusculas(texto)
  texto = separar_numeros(texto)
  texto = remplazar_ansi(texto)
  return texto

def eliminar_stopwords(texto):
    texto_limpio = ''
    for word in texto.split():
        if word in language_stopwords or word in non_words:
            continue
        else:
            texto_limpio += word + ' '
    return texto_limpio

def eliminar_punctuation(texto):
    for word in non_words:
        texto = texto.replace(word, '')
    return texto

def process_file(texto):
    texto_limpio = texto.lower()
    texto_limpio = eliminar_punctuation(texto_limpio)
    texto_limpio = eliminar_stopwords(texto_limpio)
    return texto_limpio
    
def limpiar_texto(texto):
  texto = re.sub(r'\W', ' ', str(texto))          #caracter especial
  texto = re.sub('\[[^]]*\]', ' ', texto)
  texto = re.sub(r'\s+[a-zA-Z]\s+', ' ', texto)   #letras solas
  texto = re.sub(r'\s+', ' ', texto, flags=re.I)  #vacios
  texto = texto.lower()                           #todo miniscula
  return texto

def filtrar_stopword_digitos(tokens):
  return [token for token in tokens if token not in STOPWORDS and not token.isdigit()]

def stem_palabras(tokens):
  return [stemmer.stem(token) for token in tokens]  

def comparar_texto(textoA,textoB):
  X = vectorizer.fit_transform([textoA,textoB])
  matriz_similar = cosine_similarity(X,X)
  return matriz_similar[0][1]

def token_stop_stem(texto):
  texto = tokenizer.tokenize(texto)
  texto = filtrar_stopword_digitos(texto)
  texto = stem_palabras(texto)
  texto = ' '.join(texto)
  return texto
  
def contar_errores(texto):
  aux = []
  cont = 0
  for i in range(len(texto)):
    aux.append(spell(texto[i]))
  for i in range(len(texto)):
    if(aux[i] != texto[i]):
      cont = cont + 1
  return (cont*100)/len(texto)

def analisis_sentimiento(texto):
  sentimiento = analyzer.polarity_scores(texto)
  return sentimiento['compound']

def similar(a, b):
  return SequenceMatcher(None, a, b).ratio()

def buscar_con_encabezado(texto):
  num = 0
  q = texto
  results = consulta_titulo(q)
  textoA = "" 
  textoB = ""
  textoC = ""
  porcentaje = 0
  maxi1 = 0
  maxi2 = 0
  link_noticia = ""
  titulo_noticia = ""
  exs = ["youtube.com","pdf","PDF","Pdf","/doc","facebook.com","instagram","linode","wikipedia","github","send_file","groups.google","pagina-web-en-mantenimiento","tadadelivery","/download","dclm","lavision","m.es.zaqs","45.79.163.254","hechosdelsureste","umss",".cu","shafaqna",".ru",".uk","twitter.com","tiktok.com","/blog",".json","bpemb","amn.bo","digitalbooks","empresarioindependiente","formulario","flacsoandes",".aspx","evacopa","kerathris","oepayml","mortgagenext","impuestos","gholbirn","esade","jdiezarnal","grupo.do","you-books","zoloshicage","vuramar","elnuevoherald","heraldonews","aulhala","digitalrepository","scielo.org","nj1015","intranet","baixardoc.com",".gob","tripadvisor",".xml",".gov",".doc","fliphtml5","iwgia.org","unesdoc",".txt"]
  for i in results:
    fl = 0 
    for ab in exs:
      if ab in i:
        fl = 1
    if(fl == 0):
      #print(i)
      if(".gob.bo" in i or "lostiempos" in i or "entel.bo" in i):
        r = requests.get(i,verify=False)
      else:
        r = consulta_url(i)
      bs = BeautifulSoup(r.text, 'html.parser')
      soup = BeautifulSoup(r.text, 'html5lib')
      titulo = limpieza_html(bs.title)
      #a perdefeccionar
      cuerpo = limpieza_html(soup.body)

      similitud = comparar_texto(q,titulo)
      porcentaje = porcentaje + similitud
 
      cuerpo = cuerpo[0:3000]
      cuerpo = cuerpo.replace("\n"," ")
      cuerpo = limpiar_texto(cuerpo)
      #print(cuerpo)
      
      if maxi2 < similitud and maxi1>maxi2:
        maxi2 = similitud    #0.6
        textoB = cuerpo

      if maxi1 < similitud:   #0.6
        link_noticia = i
        titulo_noticia = titulo
        textoB = textoA
        textoA = cuerpo
        maxi2 = maxi1
        maxi1 = similitud

  #print("t1 : ",textoA)
  #print("t2 : ",textoB)
  return (round(porcentaje,2),round(maxi1,2),textoA,textoB,link_noticia, titulo_noticia)

def inversa_imagen(texto):
  params = {
    "api_key": "a0740b2d83a48f405d2081009fc2c75ab7a5486ee62d87366b4b26bfd8b46260",
    "device": "desktop",
    "engine": "google_reverse_image",
    "google_domain": "google.com.mx",
    "image_url": texto,
    "gl": "mx",
    "hl": "es"
  }
  search = GoogleSearch(params)
  results = search.get_dict()
  #profile = results["knowledge_graph"]
  #print("descripcion", profile["description"])
  #print("nombre", profile["source"]["name"])
  #print("title", profile["title"])
  image_results = results["image_results"]
  return image_results

def procesar(texto):
    if(".com" not in texto):
        a1 = 0
        a2 = 0
        a3 = 0
        b1, b2, t1, t2, n1, n2 = buscar_con_encabezado(texto)
        t3 = t1
        if len(n2) > 1:
            b3 = round(comparar_texto(token_stop_stem(t1), t3),2)
            if len(t2) > 1: 
                b4 = round(comparar_texto(token_stop_stem(t2), t3),2)
                b5 = round(comparar_texto(token_stop_stem(t1), token_stop_stem(t2)),2)
            else: 
                b4 = 0
                b5 = 0
            if analisis_sentimiento(texto) > 0 and analisis_sentimiento(n2)<=0: b6 = 0
            else: b6 = 1
        else:
            b3 = 0
            b4 = 0
            b5 = 0
            b6 = 0
            #print("similitud entre los 2 primeros : ",round(comparar_texto(textoA,textoB),2))
            #print("Porcentaje suma entre los titulos : ",round(porcentaje,2))
            #print("Noticia mas similar: ",round(maxi,2))
            #print("Sentimiento :", a)
            #print("Similitud entre A y C",round(comparar_texto(textoC,textoA),2))
            #print("Similitud entre B y C",round(comparar_texto(textoC,textoB),2))
            #print("*--*",0)
        print(n1)
        print(n2)
        ar = np.array([[a1,a2,a3,b1,b2,b3,b4,b5,b6]])
        return n1,n2,ar 
    else:
        session = requests.Session()
        headers = {'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.106 Safari/537.36 OPR/38.0.2220.41'}

        q = texto

        r = session.get(q,headers=headers)
        soup = BeautifulSoup(r.text, 'html5lib')
        titulo = limpieza_html(soup.title)
        #print(titulo)
        noticia = soup.find_all("div",{"class":"container"})
        cuerpo = soup.find_all("div",{"class":"article-body"})
        #print(lim_html(cuerpo[0]))
        errores = contar_errores(limpieza_html(cuerpo[0]))
        #print("Errores ",errores)
        img_exist = 0
        try:
            img = soup.find("img",{"class":"thumbnail-img"})["src"]
            print(img)
            inversa_img = inversa_imagen(img)
        except AttributeError:
            img_exist = 1
        aux = 0
        similitud = 0
        sim = 0
        if(img_exist == 0):
            for op in inversa_img:
                aux = aux + 1
                if(aux >= 3):
                    #print(op["snippet"])
                    #print(op["title"])
                    b = comparar_texto(titulo,op["title"])
                    #print(b)
                    sim = sim + b
                    if(b > 0.4 and b <= 0):
                        similitud = 1
        
            #print("similitud ",similitud)
            #print(titulo)
        a1 = errores
        a2 = img_exist
        a3 = similitud
        b1, b2, t1, t2, n1, n2 = buscar_con_encabezado(texto)
        t3 = cuerpo
        if len(n2) > 1:
            b3 = round(comparar_texto(token_stop_stem(t1), t3),2)
            if len(t2) > 1: 
                b4 = round(comparar_texto(token_stop_stem(t2), t3),2)
                b5 = round(comparar_texto(token_stop_stem(t1), token_stop_stem(t2)),2)
            else: 
                b4 = 0
                b5 = 0
            if analisis_sentimiento(titulo) > 0 and analisis_sentimiento(n2)<=0: b6 = 0
            else: b6 = 1
        else:
            b3 = 0
            b4 = 0
            b5 = 0
            b6 = 0
            #print("similitud entre los 2 primeros : ",round(comparar_texto(textoA,textoB),2))
            #print("Porcentaje suma entre los titulos : ",round(porcentaje,2))
            #print("Noticia mas similar: ",round(maxi,2))
            #print("Sentimiento :", a)
            #print("Similitud entre A y C",round(comparar_texto(textoC,textoA),2))
            #print("Similitud entre B y C",round(comparar_texto(textoC,textoB),2))
            #print("*--*",0)
            print(n1)
            print(n2)
            ar = np.array([[a1,a2,a3,b1,b2,b3,b4,b5,b6]])
        return n1,n2,ar