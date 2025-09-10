import requests

url = "http://192.168.32.43:2000" # for robot raspberrypi

def start():

 try:
    urlstart = f"{url}/start"
    response = requests.post(urlstart)
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")
 except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")

 

def stop():
 try:
    urlstop = f"{url}/stop"
    response = requests.post(urlstop)
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")
 except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")



    
