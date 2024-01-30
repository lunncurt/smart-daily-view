import tkinter as tk
from tkinter import font
from datetime import datetime as dt
import requests
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from PIL import Image, ImageTk
from io import BytesIO
import cv2
import os
import importlib.util
from threading import Thread
import time
import numpy as np

WEATHER_API_KEY = open('weather_api_key.txt', 'r').read()
MAPS_API_KEY = open('gmaps_api_key.txt', 'r').read()

class VideoStream:
    def __init__(self,resolution=(640,480),framerate=30):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])

        (self.grabbed, self.frame) = self.stream.read()

        self.stopped = False

    def start(self):
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

def obj_det(tk_frame, tk_label):
    #Checking if this is the Pi or Laptop
    pi_check = importlib.util.find_spec('tflite_runtime')
    if pi_check:
        from tflite_runtime.interpreter import Interpreter
    else:
        from tensorflow.lite.python.interpreter import Interpreter

    LABEL_FILE = 'labelmap.txt'
    TFLITE_INTERPRETER = 'detect_quant.tflite'
    MODEL_DIR = 'saved_model'

    CWD_PATH = os.getcwd()
    LBL_PATH = os.path.join(CWD_PATH, MODEL_DIR, LABEL_FILE)
    INTR_PATH = os.path.join(CWD_PATH, MODEL_DIR, TFLITE_INTERPRETER)

    interpreter = Interpreter(model_path=INTR_PATH)
    interpreter.allocate_tensors()

    MIN_THRESHOLD = 0.75
    RESX = 640
    RESY = 360

    with open(LBL_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    boxes_idx, classes_idx, scores_idx = 1, 3, 0

    fps = 1
    freq = cv2.getTickFrequency()

    videostream = VideoStream(resolution=(RESX, RESY), framerate=30).start()
    time.sleep(1)

    confidence_threshold_duration = 2.0
    quiet_period_duration = 240.0
    object_name_in_focus = None
    start_time_in_focus = None
    last_detection_time = time.time() - quiet_period_duration

    while True:
        t1 = cv2.getTickCount()

        frame = videostream.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

        # Find the index of the detection with the highest confidence score
        max_score_index = np.argmax(scores)

        if ((scores[max_score_index] > MIN_THRESHOLD) and (scores[max_score_index] <= 1.0)):
            ymin, xmin, ymax, xmax = boxes[max_score_index]
            ymin, xmin, ymax, xmax = int(ymax * RESY), int(xmin * RESX), int(ymin * RESY), int(xmax * RESX)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            object_name = labels[int(classes[max_score_index])]
            label = f'{object_name}: {int(scores[max_score_index] * 100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            detected_object = object_name

            if detected_object == object_name_in_focus and start_time_in_focus is not None:
                duration_in_focus = time.time() - start_time_in_focus
                if duration_in_focus >= confidence_threshold_duration:
                    if time.time() - last_detection_time >= quiet_period_duration:
                        detection_results = f"'{object_name_in_focus}' detected for {confidence_threshold_duration} seconds."
                        last_detection_time = time.time()
                        update_info(object_name_in_focus)
            else:
                object_name_in_focus = detected_object
                start_time_in_focus = time.time()

        # Convert the frame to a PhotoImage
        tk_frame.image = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        tk_label.config(image=tk_frame.image)

        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        fps = 1 / time1

        if cv2.waitKey(1) == ord('q'):
            terminate()
            break

    cv2.destroyAllWindows()
    videostream.stop()

def obj_det_thread():
    obj_det(video_frame, video_label)

def update_time():
    current_time = dt.now().strftime("%I:%M:%S %p")
    current_day = (dt.now().strftime("%A, %B %d")
                   .replace(str(dt.now().day), str(dt.now().day)
                                + ("th" if 10 <= dt.now().day % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}
                                    .get(dt.now().day % 10, 'th'))))

    date_label.config(text = current_day)
    time_label.config(text = current_time)

    display_window.after(1000, update_time)

def update_info(person = "House"):
    with open('calendar_ids.txt', 'r') as file:
        lines = file.readlines()

    if person == "curtis":
        id = lines[0].strip()
    elif person == "darian":
        id = lines[1].strip()
    elif person == "jake":
        id = lines[2].strip()
    elif person == "matt":
        id = lines[3].strip()
    else:
        id = "primary"

    calendar_label.config(text = calendar(person, id))
    weather_label.config(text = get_weather())

def get_weather():
    BASE_URL = "https://api.openweathermap.org/data/3.0/onecall?lat=42.73&lon=-84.48&units=imperial&appid="
    url = BASE_URL + WEATHER_API_KEY

    results = requests.get(url).json()

    temp = (    '\n  Temperature:  ' + str(int(results['current']['temp'])) + '\u00b0F (feels like '
                + str(int(results['current']['feels_like']))+ '\u00b0)')
    high =      '\n    High: ' + str(int(results['daily'][0]['temp']['max'])) + '\u00b0F'
    low =       '\n    Low: ' + str(int(results['daily'][0]['temp']['min'])) + '\u00b0F'
    condition = '\n  Condition: ' + results['current']['weather'][0]['description']
    summary =   '\n  Summary: ' + results['daily'][0]['summary']

    weather = 'Here is todays weather report:' + temp + high + low + condition + summary

    return weather

def get_ttime(destination, dep_time, eventSum):
    origin = "253 Milford Street, East Lansing, Michigan."

    current_timestamp = int(time.time())
    if dep_time is not None and dep_time <= current_timestamp:
        dep_time = current_timestamp

    directions_url = "https://maps.googleapis.com/maps/api/directions/json"
    directions_params = {
        "key": MAPS_API_KEY,
        "origin": origin,
        "destination": destination,
        "mode": "driving",
        "traffic_model": "best_guess",
        "departure_time": dep_time
    }

    directions_response = requests.get(directions_url, params=directions_params)
    directions_data = directions_response.json()

    if 'routes' in directions_data and directions_data['routes']:
        travel_time = directions_data['routes'][0]['legs'][0]['duration']['text']
        get_timage(directions_data, destination)

        ttime_label.config(text=str(travel_time) + " estimated travel time to: " + eventSum)
    else:
        ttime_label.config(text="No routes found")

def get_timage(data, dest):
    ppoints = data['routes'][0]['overview_polyline']['points']
    origin = "253 Milford Street, East Lansing, Michigan"

    static_url = "https://maps.googleapis.com/maps/api/staticmap"
    static_params = {
        "key": MAPS_API_KEY,
        "size": "600x400",
        "path": f"enc:{ppoints}",
        "mode": "driving",
        "markers": f"color:red|label:A|{origin}",
        "markers": f"color:red|label:O|{dest}"
    }

    static_response = requests.get(static_url, params=static_params)

    if static_response.status_code == 200:
        image_content = Image.open(BytesIO(static_response.content))
        tk_image = ImageTk.PhotoImage(image_content)

        ttime_image.config(image=tk_image)
        ttime_image.image = tk_image
    else:
        print(f"Error: {static_response.status_code}, {static_response.text}")

def calendar(person, calendar_ID = "primary"):
    SCOPES = ["https://www.googleapis.com/auth/calendar"]
    calout = 'Hello ' + person + '! Heres your schedule for today:\n'
    destination_out = None

    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    try:
        service = build("calendar", "v3", credentials=creds)

        now = dt.now()

        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        time_min = start_of_day.isoformat() + "Z"

        end_of_day = now.replace(hour=23, minute=59, second=59, microsecond=999999)
        time_max = end_of_day.isoformat() + "Z"

        events_result = (
            service.events()
            .list(
                calendarId=calendar_ID,
                timeMin=time_min,
                timeMax=time_max,
                maxResults=5,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        events = events_result.get("items", [])

        if not events:
            calout += '   -Nothing on the schedule for today!'
            get_ttime(destination_out, None, None)
            return calout

        for i, event in enumerate(events):
            start = event["start"].get("dateTime")
            start_datetime = dt.fromisoformat(start)
            start_timestamp = int(start_datetime.timestamp())
            start_time = dt.fromisoformat(start).strftime("%I:%M %p")
            calout += "   - " + event["summary"] + " at " + start_time

            if i == 0:
                destination_out = event.get('location', 'Location not available')
                get_ttime(destination_out, start_timestamp, event["summary"])


            if i < len(events) - 1:
                calout += '\n'

        return calout

    except HttpError as error:
        print(f"An error occurred: {error}")
        quit()

def terminate():
    display_window.destroy()

display_window = tk.Tk()
display_window.geometry('1920x1080')
#display_window.attributes('-fullscreen', True)
display_window.title("Day At a Glance")

time_label = tk.Label(display_window)
time_label.place(x = 550, y = 30)
time_label.configure(font = tk.font.Font(family = "Courier New", size = 80, weight='bold'))

date_label = tk.Label(display_window)
date_label.place(x = 400, y = 120)
date_label.configure(font = tk.font.Font( family = "Courier New", size = 60))

weather_label = tk.Label(display_window, wraplength=1000, justify='left')
weather_label.place(x = 20, y = 200)
weather_label.configure(font = tk.font.Font(family = "Courier New", size = 35))

calendar_label = tk.Label(display_window, justify='left')
calendar_label.place(x = 20, y = 600)
calendar_label.configure(font = tk.font.Font(family = "Courier New", size = 35))

ttime_image = tk.Label(display_window)
ttime_image.place(x=1300, y=500)

ttime_label = tk.Label(display_window, wraplength=500, justify='left')
ttime_label.place(x = 1300, y = 400)
ttime_label.configure(font = tk.font.Font(family = "Courier New", size = 35))

video_frame = tk.Frame(display_window, width=300, height=200)
video_frame.place(x=1260, y=0)

video_label = tk.Label(video_frame)
video_label.pack()

update_time()
update_info()

obj_det_thread = Thread(target=obj_det_thread)
obj_det_thread.start()

display_window.mainloop()

#Use to show all fonts available for selection
#print(font.families())