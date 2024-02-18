# smart-daily-view
This project was designed to give my roommates and myself a daily view screen in our kitchen that would emphasize the most important information for the day at a glance.

## How it works

### User-Specific Info
- The USB Webcam connected to the Pi using OpenCV pulls in and displays a live video feed, that is also put into a custom Tensorflow Lite model trained to distinguish our different faces.
- Once a user is detected, it displays their upcoming events for the day by accessing Google Calendars API.
- This event's location is then passed into Google Maps API to calculate the travel time from the house and display the actual route.
- If no user is detected, it will display the next events for all users, and re-attempt in a few minutes.

### Weather
- Pulls weather info using Open Weathers API.
- Displays the current locations temperature (high, low, and current), weather condition, and a quick summary of the days expected conditions.
