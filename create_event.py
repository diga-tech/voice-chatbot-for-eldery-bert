import datetime 
import os.path
from gtts import gTTS
import os
import speech_recognition as sr
import re
import dateparser
import pytz
import time
import Jetson.GPIO as GPIO

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

audio_response = "mpg123 -a plughw:3,0 buatkegiatan.mp3" #for linux
# audio_response = "start res.mp3" #for windows

# Define the GPIO pin for the LED
LED_PIN = 18

# Set up the GPIO mode
GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering
GPIO.setup(LED_PIN, GPIO.OUT)  # Set the LED pin as output

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/calendar"]


def parse_datetime(date_part, time_part, timezone='Asia/Jakarta', language='id'):
    # Combine the date and time parts
    datetime_str = f"{date_part} {time_part}"
    
    # Parse the combined datetime string
    parsed_dt = dateparser.parse(datetime_str, languages=[language])
    
    if parsed_dt is None:
        # raise ValueError("Could not parse date and time from the input.")
        print("Gagal Menyimpan Ulangi kembali")
        tts = gTTS(text="Gagal Menyimpan Ulangi kembali", lang = 'id')
        tts.save("buatkegiatan.mp3")
        os.system(audio_response)
        time.sleep(10)
        create_event()
        return None

    # Set the timezone
    local_tz = pytz.timezone(timezone)
    localized_dt = local_tz.localize(parsed_dt)
    
    # Convert to ISO 8601 format with timezone offset
    iso_format = localized_dt.isoformat()
    return iso_format

def suara():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Silakan bicara dalam Bahasa Indonesia...")
        GPIO.output(LED_PIN, GPIO.HIGH)
        audio = recognizer.listen(source)
    try:
        # Convert speech to text using Google Web Speech API with Indonesian language ('id-ID')
        text = recognizer.recognize_google(audio, language="id-ID")
        GPIO.output(LED_PIN, GPIO.LOW)
        print(f"Teks yang dikenali: {text}")
    except sr.UnknownValueError:
        text = "Maaf, saya tidak bisa mengenali ucapan."
    return text

def create_event():
  """Shows basic usage of the Google Calendar API.
  Prints the start and name of the next 10 events on the user's calendar.
  """
  
  creds = None
  # The file token.json stores the user's access and refresh tokens, and is
  # created automatically when the authorization flow completes for the first
  # time.
  if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          "data/credentials.json", SCOPES
      )
      creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open("token.json", "w") as token:
      token.write(creds.to_json())

  try:
    service = build("calendar", "v3", credentials=creds)
 
    ## Nama Kegiatan
    print("Kegiatan apa")
    tts = gTTS(text="Acara Apa", lang = 'id')
    tts.save("buatkegiatan.mp3")
    os.system(audio_response)
    time.sleep(2)
    event_texts = suara()

    ## Tanggal Kegiatan
    print("Tanggal Berapa")
    tts = gTTS(text="Tanggal Berapa", lang = 'id')
    tts.save("buatkegiatan.mp3")
    os.system(audio_response)
    time.sleep(2)
    tanggal = suara()

    ## jam Kegiatan
    print("Jam Berapa")
    tts = gTTS(text="Jam Berapa", lang = 'id')
    tts.save("buatkegiatan.mp3")
    os.system(audio_response)
    time.sleep(2)
    jam = suara()
    jam = jam.replace("jam", "").strip() + ":00" 
    parsed_date = parse_datetime(tanggal, jam)
    

    event = {
        'summary': event_texts,
        'start': {
            'dateTime': parsed_date,
            'timeZone': 'Asia/Jakarta',
        },
        'end': {
            'dateTime': parsed_date,
            'timeZone': 'Asia/Jakarta',
        },
    }

    events = service.events().insert(calendarId='primary', body=event).execute()
    print('Event created: %s' % (events.get('htmlLink')))


  except HttpError as error:
    print(f"An error occurred: {error}")

  return "Kegiatan telah disimpan"

if __name__ == "__main__":
  create_event()
