import tensorflow as tf
import random
import json
import string
import re
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from create_event import create_event
from list_event import list_event
from robot import start, stop
import pandas as pd
import datetime
import nlp_id
from nlp_id.lemmatizer import Lemmatizer
from nlp_id.stopword import StopWord
import pytz
import requests
import speech_recognition as sr
from gtts import gTTS
import os
from mutagen.mp3 import MP3
import time
from openpyxl import Workbook
import Jetson.GPIO as GPIO

import threading
from flask import Flask, jsonify
app = Flask(__name__)



# Define the GPIO pin for the LED
LED_PIN = 18

# Set up the GPIO mode
GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering
GPIO.setup(LED_PIN, GPIO.OUT)  # Set the LED pin as output

audio_response = "mpg123 -a plughw:3,0 res.mp3" #for linux
# audio_response = "start res.mp3" #for windows

# Speech to Text
recognizer = sr.Recognizer()
def listen_and_convert_to_text():
    
    with sr.Microphone() as source:
        print("Adjusting for ambient noise, please wait...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Silakan bicara dalam Bahasa Indonesia...")
        GPIO.output(LED_PIN, GPIO.HIGH)
        audio = recognizer.listen(source)

    try:
        GPIO.output(LED_PIN, GPIO.LOW)
        print("Mengenali ucapan...")
        # Convert speech to text using Google Web Speech API with Indonesian language ('id-ID')
        text = recognizer.recognize_google(audio, language="id-ID")
        print(f"Teks yang dikenali: {text}")
        return text
    except sr.UnknownValueError:
        text = "Maaf, saya tidak bisa mengenali ucapan."
        return text
    

#API Cuaca
def get_weather(adm_code="32.73.02.1005"):
    
    # URL API BMKG untuk prakiraan cuaca
    url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={adm_code}"

    try:
        # Mengirim permintaan ke API
        response = requests.get(url)
        response.raise_for_status()  # Memastikan permintaan berhasil

        # Mengambil data dalam format JSON
        data = response.json()
        cuaca = data['data'][0]['cuaca'][0][1]['weather_desc']
        suhu = data['data'][0]['cuaca'][0][1]['t']
        kelembapan = data['data'][0]['cuaca'][0][1]['hu']

        # Menyusun hasil dalam dictionary
        weather_info = {
            "cuaca": cuaca,
            "suhu": suhu,
            "kelembapan": kelembapan
        }

        return f"{weather_info['cuaca']} dengan suhu {weather_info['suhu']} dan kelembapan {weather_info['kelembapan']}"
    except requests.exceptions.RequestException as e:
        print("Terjadi kesalahan saat mengambil data cuaca:", e)



def get_news():
    # URL API NEWS
    url = f"https://berita-indo-api-next.vercel.app/api/tempo-news/tekno"
    try:
            # Mengirim permintaan ke API
        response = requests.get(url)
        response.raise_for_status()  # Memastikan permintaan berhasil

            # Mengambil data dalam format JSON
        data = response.json()
        news = data['data'][1]['title']       

        return news
    except requests.exceptions.RequestException as e:
        print("Terjadi kesalahan saat mengambil data berita", e)



# API function
def real_time():
    today = datetime.datetime.now(pytz.timezone('Asia/Jakarta'))
    return today

# API Sholat
def get_prayer_times(city, country, method, x):
    url = f"http://api.aladhan.com/v1/timingsByCity?city={city}&country={country}&method={method}"
    response = requests.get(url)
    data = response.json()

    # Dictionary of prayer times
    prayer_times = {
        "subuh": data['data']['timings']['Fajr'],
        "dzuhur": data['data']['timings']['Dhuhr'],
        "ashar": data['data']['timings']['Asr'],
        "magrib": data['data']['timings']['Maghrib'],
        "isya": data['data']['timings']['Isha']
    }
    
    # Loop to check if x matches any prayer time
    for prayer, time in prayer_times.items():
        if prayer == x:
            return {prayer: time}
    
    # If no match is found, return None
    return  prayer_times

##Stop Word
# Initialize the StopWord class
stopword = StopWord()

# Specify the word you want to remove
word_to_remove = ['sendirian', 'sendiri', 'sendirinya', 'bekerja', 'memulai', 'mulai', 'bulan', 'hari', 'minggu', 'bertanya', 'nanya', 'tanya', 'siapa', 'bikin', 'buat', 'membuat', 'jangan']

# Remove the word from the stopword list if it exists
for word in word_to_remove:
    if word in stopword.stopwords:
        stopword.stopwords.remove(word)

lemmatizer_id = Lemmatizer()

# Pre-process functions
def casefolding(x):
    x = x.lower()  # Lowercase the text
    x = re.sub(r"\d+", "", x)  # Remove numbers
    x = x.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    x = x.strip()  # Remove whitespace
    return x

def text_normalize(x, key_norms):
    x = ' '.join([key_norms[key_norms['singkat'] == word]['hasil'].values[0] if (key_norms['singkat'] == word).any() else word for word in x.split()])
    x = str.lower(x)
    return x


def stop_word_id(x):
  x = x.split()
  x = [word for word in x if word not in stopword.stopwords]
  return " ".join(x)

def lemmati(x):
  x = lemmatizer_id.lemmatize(x)
  return x

def text_cleaning(text, key_norms):
    text = casefolding(text)
    text = text_normalize(text, key_norms)
    text = stop_word_id(text)
    text = lemmati(text)

    return text
# Load the tokenizer and model
Pre_training = 'indobenchmark/indobert-lite-base-p2'
tokenizer = BertTokenizer.from_pretrained(Pre_training)
model = TFBertForSequenceClassification.from_pretrained(Pre_training, num_labels=50)

# Load the model weights (fine-tuned)
path_model = "model/modelQA.h5"
model.load_weights(path_model)


# Load the dataset and responses
with open("data/dataset_lansia.json", 'r') as json_file:
    datasetQA = json.load(json_file)

# Extract questions, labels, and responses
responses = {}
inputs = []
tags = []
for intent in datasetQA['data']:
    responses[intent['label']] = intent['answers']
    for lines in intent['questions']:
        inputs.append(lines)
        tags.append(intent['label'])

# Create label encoder
data = pd.DataFrame({"questions": inputs, "label": tags})
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

ERROR_Treshold = 0.8

def terjatuh():
    text = "Selalu gunakan alas kaki yang nyaman dan perhatikan lantai agar tidak licin. Pastikan penerangan cukup, terutama saat malam hari, untuk mengurangi risiko terjatuh."
    tts = gTTS(text=text, lang = 'id')
    tts.save("res.mp3")
    et = time.time()
    os.system(audio_response)

def save_to_excel(data, filename, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Full path to the file
    file_path = os.path.join(folder_name, filename)
    """Save the data to an Excel file with the specified filename."""
    workbook = Workbook()
    sheet = workbook.active

    # Add headers to the sheet
    sheet.append(["uji_ke", "input","clean", "label", "prediksi_response", "waktu_response", 'hasil_response'])

    # Add data rows to the sheet
    for row in data:
        sheet.append(row)

    # Save the workbook with the provided filename
    if not file_path.endswith(".xlsx"):
        file_path += ".xlsx"
    workbook.save(file_path)
    print(f"Excel file '{filename}' created and data added successfully!")

# Main chatbot loop
def chatbot_loop():
    print("Chatbot: Hello! Type 'selesai' to stop the conversation.")
    data_add = []
    uji = 0
    while True:
        # user_input = input("You: ")
        print("You:")
        uji += 1

        user_input = listen_and_convert_to_text()
        st = time.time()
        # Exit the loop if the user types 'exit'
        if user_input.lower() == "selesai":
            print("Chatbot: Goodbye!")
            break
        
        # Clean and preprocess the user input
        key_norms = pd.read_csv("data/key_norm_v2.csv")
        cleaned_input = text_cleaning(user_input, key_norms)
        
        # Tokenize the input for BERT
        input_text_tokenized = tokenizer.encode(cleaned_input,
                                                add_special_tokens=True,
                                                truncation=True,
                                                padding='max_length',
                                                return_tensors='tf')

        # Predict with the model
        bert_predict = model(input_text_tokenized)
        bert_predict = tf.nn.softmax(bert_predict[0], axis=-1)
        output = tf.argmax(bert_predict, axis=1)
        predict = bert_predict[0][output[0]].numpy()
        response_tag = le.inverse_transform([output.numpy()[0]])[0]
        print(f'Label: {response_tag}, prediksi: {predict}')
        
        today = real_time()
        date = today.strftime("%d. %B %Y")
        month = today.strftime("%B")
        day = today.strftime("%A")
        jam = today.strftime("%H:%M")
        
        
        # Get the predicted label and select a random response
        if predict > ERROR_Treshold:
            response = random.choice(responses[response_tag])
            
            if response_tag == 'tanggal':
                text=f'{response} {date}'
                print(f"Chatbot: {text}")
                tts = gTTS(text=text, lang = 'id')
                


            elif response_tag == 'hari':
    
                text=f'{response} {day}'
                print(f"Chatbot: {text}")
                tts = gTTS(text=text, lang = 'id')
                

            elif response_tag == 'bulan':
    
                text=f'{response} {month}' 
                print(f"Chatbot: {text}")
                tts = gTTS(text=text, lang = 'id')
                  
                
            elif response_tag == 'jam':
    
                text=f'{response} {jam}'
                print(f"Chatbot: {text}")
                tts = gTTS(text=text, lang = 'id')
                
                
                
            elif response_tag == 'ibadah':
                prayer = get_prayer_times("Bandung", "Indonesia",'2', 'all')
                text=f'{response} {prayer}'
                print(f"Chatbot: {text}")
                tts = gTTS(text=text, lang = 'id')
                

            elif response_tag == 'subuh':
                prayer = get_prayer_times("Bandung", "Indonesia",'2', response_tag)
                text=f'{response} {prayer}'
                print(f"Chatbot: {text}")
                tts = gTTS(text=text, lang = 'id')
                

            elif response_tag == 'dzuhur':
                
                prayer = get_prayer_times("Bandung", "Indonesia",'2', response_tag)
                text=f'{response} {prayer}'
                print(f"Chatbot: {text}")
                tts = gTTS(text=text, lang = 'id')
            
            elif response_tag == 'ashar':
                prayer = get_prayer_times("Bandung", "Indonesia",'2', response_tag)
                text=f'{response} {prayer}'
                print(f"Chatbot: {text}")
                tts = gTTS(text=text, lang = 'id')
                

            elif response_tag == 'magrib':
                prayer = get_prayer_times("Bandung", "Indonesia",'2', response_tag)
                text=f'{response} {prayer}'
                print(f"Chatbot: {text}")
                tts = gTTS(text=text, lang = 'id')
                

            elif response_tag == 'isya':
                prayer = get_prayer_times("Bandung", "Indonesia",'2', response_tag)
                text=f'{response} {prayer}'
                print(f"Chatbot: {text}")
                tts = gTTS(text=text, lang = 'id')
                
            elif response_tag == 'membuatkegiatan':
                text = response
                print(f"Chatbot: {text}")
                tts = gTTS(text=text, lang = 'id')
                tts.save("res.mp3")
                os.system(audio_response)
                create_event()
                time.sleep(10)

            elif response_tag == 'daftarkegiatan':
                res = list_event()
                text=f'{response} {res}'
                print(f"Chatbot: {text}")
                tts = gTTS(text=text, lang = 'id')

            elif response_tag == 'mulai':
                text=f'{response}'
                start()
                print(f"Chatbot: {text}")
                tts = gTTS(text=text, lang = 'id')

            elif response_tag == 'berhenti':
                text=f'{response}'
                stop()
                print(f"Chatbot: {text}")
                tts = gTTS(text=text, lang = 'id')


            elif response_tag == 'cuaca':
                weather_data = get_weather()
                text=f'{response} {weather_data}'
                print(f"Chatbot: {text}")
                tts = gTTS(text=text, lang = 'id')

            elif response_tag == 'berita':
                news = get_news()
                text=f'{response} {news}'
                print(f"Chatbot: {text}")
                tts = gTTS(text=text, lang = 'id')
		
            elif response_tag == 'ulang':
                text=f'{response} {save_resp}'
                print(f"Chatbot: {text}")
                tts = gTTS(text=text, lang = 'id')
                
            else:
                text = response
                print(f"Chatbot: {text}")
                tts = gTTS(text=text, lang = 'id')
                
                
        else:
            text = "Saya tidak mengerti"
            print(f"Chatbot: {text}")
            tts = gTTS(text=text, lang = 'id')
            
        tts.save("res.mp3")
        et = time.time()
        os.system(audio_response) 
        audio = MP3("res.mp3")
        duration = audio.info.length
        #time.sleep(duration)
        res_time= et - st
        save_resp = text
        print(f"waktu yang dibutuhkan {res_time} second")
        data_add.append([uji, user_input, cleaned_input, response_tag, predict, res_time, text])
        print("Data added!")
    return data_add

# Start the chatbot loop
def  run_chatbot():
  data = chatbot_loop()
  subjek_filename = input("Enter the file name (with .xlsx extension): ")
  save_to_excel(data, subjek_filename, "data_collect")


@app.route("/fall", methods=["POST"])
def fall():
  terjatuh()
  return jsonify({"status": "success", "message": "Lansia Terjatuh"}), 200

if __name__ == "__main__":
    chatbot_thread = threading.Thread(target=run_chatbot, daemon=True)
    chatbot_thread.start()
    app.run(host="0.0.0.0", port=7000)
