## ProActRail
ProActRail is a Streamlit application for monitoring and predicting equipment failure in railway systems. It simulates and visualizes real-time data, including sensor readings and GPS coordinates, to detect potential oil leaks and other issues. The application uses a machine learning model to predict equipment failure and sends alerts to help prevent accidents.

# Features

- Real-time data simulation and visualization
- Oil leak detection and alerts
- Machine learning model for failure prediction
- Integration with BigQuery for data storage and retrieval
- Integration with Google Cloud Pub/Sub for data streaming

# Installation

Clone the repository:
```bash
git clone https://github.com/9elmo6/ProActRail.git
cd ProActRail
```
Create a virtual environment and activate it:
```bash

python3 -m venv venv
source venv/bin/activate
```

Install the required packages:
```bash
pip install -r requirements.txt
```
# Usage

Set up your Google Cloud credentials by following the instructions in the Google Cloud documentation.
Export the path to your Google Cloud credentials JSON file:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/credentials.json"
```
Run the Streamlit application:
```bash
streamlit run streamlit-app.py
```

Open the provided URL in your web browser to interact with the application.

# To-DO:
- Add Email notification in case of failure detected
- Add Air leak simulation
- Visualize the model predictions on the plot




# Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

License
[MIT](https://choosealicense.com/licenses/mit/)

