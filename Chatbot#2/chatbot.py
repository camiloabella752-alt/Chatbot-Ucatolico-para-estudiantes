# ================== Librerías necesarias ==================

# Librería estándar para generar valores aleatorios (se usa en la lógica del bot)
import random

# Librería para manejar archivos en formato JSON (donde tienes los intents y respuestas)
import json

# Librería de Python para guardar/cargar objetos serializados (como listas o diccionarios)
import pickle

# Librería NumPy para operaciones matemáticas y manejo de arreglos
import numpy as np

# NLTK (Natural Language Toolkit), una librería para procesamiento de lenguaje natural
import nltk

# Importamos el lematizador de WordNet desde NLTK
# Este se usa para reducir palabras a su forma base (ejemplo: "running" → "run")
from nltk.stem import WordNetLemmatizer

# Desde Keras importamos load_model, para cargar un modelo de red neuronal ya entrenado
from keras.models import load_model


# ================== Inicialización ==================

# Creamos una instancia del lematizador de WordNet
lemmatizer = WordNetLemmatizer()

# Abrimos y cargamos el archivo JSON con los intents (intenciones del usuario)
# El parámetro encoding='utf-8' asegura que se lean bien los acentos y caracteres especiales
intents = json.loads(
    open('intents_spanish.json', 'r', encoding='utf-8').read())


# ================== Cargar archivos previamente entrenados ==================

# Cargamos el archivo 'words.pkl', que contiene el vocabulario de palabras procesadas
words = pickle.load(open('words.pkl', 'rb'))

# Cargamos el archivo 'classes.pkl', que contiene las categorías (intents) a las que puede responder el chatbot
classes = pickle.load(open('classes.pkl', 'rb'))

# Cargamos el modelo entrenado (chatbot_model.h5), que es una red neuronal guardada con Keras
model = load_model('chatbot_model.h5')


# ================== Función para limpiar la oración del usuario ==================

def clean_up_sentence(sentence):
    """
    Esta función recibe una oración (texto de entrada del usuario)
    y la procesa para que el modelo pueda entenderla.
    """
    # Tokenizamos la oración: dividimos la frase en palabras individuales
    # Ejemplo: "Hola, ¿cómo estás?" → ["Hola", ",", "¿", "cómo", "estás", "?"]
    sentence_words = nltk.word_tokenize(sentence)
    # Aplicamos lematización: reducimos cada palabra a su forma base y la pasamos a minúsculas
    # Ejemplo: "Running" → "run", "Dogs" → "dog"
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    # Retornamos la lista de palabras procesadas
    return sentence_words


def bag_of_words(sentence):
    # Convertimos la oración del usuario en una "bolsa de palabras"
    # (vector binario donde cada posición indica si una palabra del vocabulario aparece o no en la oración)

    sentence_words = clean_up_sentence(sentence)
    # Llamamos a la función anterior para tokenizar y lematizar la oración
    # Ejemplo: "Hola, ¿cómo estás?" → ["hola", "cómo", "estar"]

    bag = [0] * len(words)
    # Creamos una lista de ceros del mismo tamaño que nuestro vocabulario (words.pkl)
    # Ejemplo: vocabulario = ["hola", "adiós", "gracias"] → bag = [0, 0, 0]

    for w in sentence_words:
        # Recorremos las palabras que escribió el usuario
        for i, word in enumerate(words):
            # Recorremos todo el vocabulario
            if word == w:
                # Si la palabra del usuario coincide con una palabra del vocabulario...
                bag[i] = 1
                # ...ponemos un 1 en esa posición
                # Ejemplo: "hola" → [1, 0, 0]

    return np.array(bag)
    # Retornamos el vector final como un arreglo de NumPy


def predict_class(sentence):
    # Esta función predice a qué intención (intent) pertenece la oración del usuario

    bow = bag_of_words(sentence)
    # Convertimos la oración en bolsa de palabras

    res = model.predict(np.array([bow]))[0]
    # Usamos el modelo de red neuronal para predecir la probabilidad de cada intent
    # Devuelve algo como: [0.1, 0.7, 0.05, 0.15] → (70% para intent 2)

    ERROR_THRESHOLD = 0.25
    # Definimos un umbral mínimo de confianza (25%)
    # Si la probabilidad es menor, lo descartamos

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Filtramos solo los intents con probabilidad mayor al umbral
    # Ejemplo: si intent 1 = 0.7 y los demás < 0.25 → results = [[1, 0.7]]

    results.sort(key=lambda x: x[1], reverse=True)
    # Ordenamos los resultados de mayor a menor probabilidad

    return_list = []
    # Creamos una lista vacía para almacenar los intents válidos

    for r in results:
        # Recorremos cada resultado válido
        return_list.append({
            'intent': classes[r[0]],       # Guardamos el nombre del intent
            # Guardamos la probabilidad en string
            'probability': str(r[1])
        })

    return return_list
    # Retornamos la lista con las predicciones


def get_response(intents_list, intents_json):
    # intents_list: lista de intenciones predichas por el modelo (probabilidad + intent)
    # intents_json: archivo JSON con las intenciones y posibles respuestas

    tag = intents_list[0]['intent']
    # Obtiene la etiqueta (tag) de la primera intención más probable detectada por el modelo.

    list_of_intents = intents_json['intents']
    # Extrae del JSON la lista de intenciones definidas (cada intent tiene un 'tag' y 'responses').

    # Recorre cada intención del JSON
    for i in list_of_intents:
        # Compara si el tag de la predicción coincide con el tag del JSON
        if i['tag'] == tag:
            # Selecciona aleatoriamente una respuesta entre las posibles respuestas del intent
            result = random.choice(i['responses'])

            # Rompe el bucle porque ya encontró la intención correcta
            break

    return result
    # Retorna la respuesta seleccionada
