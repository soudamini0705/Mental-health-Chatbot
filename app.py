import streamlit as st
import nltk
nltk.download('popular')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

# Load chatbot model
model = load_model("model.h5")

# Initialize NLTK
lemmatizer = WordNetLemmatizer()

# Load intents, words, classes
intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

# Huggingface translators
eng_swa_tokenizer = AutoTokenizer.from_pretrained("Rogendo/en-sw")
eng_swa_model = AutoModelForSeq2SeqLM.from_pretrained("Rogendo/en-sw")
eng_swa_translator = pipeline("text2text-generation", model=eng_swa_model, tokenizer=eng_swa_tokenizer)

swa_eng_tokenizer = AutoTokenizer.from_pretrained("Rogendo/sw-en")
swa_eng_model = AutoModelForSeq2SeqLM.from_pretrained("Rogendo/sw-en")
swa_eng_translator = pipeline("text2text-generation", model=swa_eng_model, tokenizer=swa_eng_tokenizer)

def translate_text_eng_swa(text):
    return eng_swa_translator(text, max_length=128, num_beams=5)[0]['generated_text']

def translate_text_swa_eng(text):
    return swa_eng_translator(text, max_length=128, num_beams=5)[0]['generated_text']

# Spacy Language Detector
def get_lang_detector(nlp, name):
    return LanguageDetector()

nlp = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe('language_detector', last=True)

# Chatbot functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def getResponse(ints, intents_json):
    if ints:
        tag = ints[0]['intent']
        for i in intents_json['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return "Sorry, I didn't understand that."

def chatbot_response(msg):
    doc = nlp(msg)
    detected_language = doc._.language['language']

    if detected_language == "en":
        res = getResponse(predict_class(msg, model), intents)
        return res
    elif detected_language == 'sw':
        translated_msg = translate_text_swa_eng(msg)
        res = getResponse(predict_class(translated_msg, model), intents)
        return translate_text_eng_swa(res)
    else:
        return "Unsupported language."

# ===============================
# STREAMLIT UI
# ===============================
st.title("🧠 Mental Health Chatbot (EN ↔ SW)")

user_input = st.text_input("Type your message:")

if st.button("Send"):
    if user_input.strip():
        bot_response = chatbot_response(user_input)
        st.write(f"**Bot:** {bot_response}")
    else:
        st.write("⚠️ Please enter a message.")

