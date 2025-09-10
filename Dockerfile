FROM python:3.11.10-bullseye

WORKDIR /app

COPY . .

RUN apt-get update
RUN apt-get install alsa-utils libasound2-dev libportaudio2 libportaudiocpp0 portaudio19-dev flac nano mpg123 -y
RUN pip install -r requirements.txt
RUN pip install pyaudio tf-keras Jetson.GPIO nlp-id flask
RUN pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib

COPY . .

CMD [ "python", "app.py" ]