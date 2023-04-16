import streamlit as st
import pandas as pd
import numpy as np
import time
import altair as alt
import io
import time
import random
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objs as go

from google.cloud import bigquery
from google.cloud import pubsub_v1
from google.cloud import bigquery
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from data_pre import preprocess_data
import plotly.express as px # interactive charts 
st.set_page_config(
    page_title = 'Real-Time Data from APU',
    page_icon = '✅',
    layout = 'wide'
)

model = load_model("models/first.h5")

st.title('Predictive maintenance dashboard')
DATE_COLUMN = 'timestamp'
DATA_URL = ('data/dataset_train.csv')

project_id = "Add Project name" #change here
topic_id = "Add Pub/sub topic" #change here

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_id)

#@st.cache_resource 
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data
data = load_data(10000)

def get_data_from_bigquery():
    client = bigquery.Client()
    query = """
        SELECT *
        FROM `project-id.Dataset.tablename`
        
        ORDER BY timestamp DESC
        LIMIT 100
    """
    query_job = client.query(query)
    data = query_job.result().to_dataframe()
    return data
def generate_live_data(column, n_points=1):
    mean = data[column].mean()
    std = data[column].std()
    new_data = np.random.normal(mean, std, n_points)
    if column in ['comp', 'dv_electric', 'towers', 'mpg', 'lps', 'pressure_switch','dv_eletric', 'oil_level', 'caudal_impulses', 'gps_speed', 'gps_quality']:
        return np.round(new_data).astype(int)
    else:
        return new_data

def get_filtered_data_from_bigquery(start_date, end_date):
    client = bigquery.Client()
    query = f"""
        SELECT *
        FROM `project-id.Dataset.tablename`
        WHERE timestamp >= '{start_date}' AND timestamp <= '{end_date}'
        ORDER BY timestamp
    """
    query_job = client.query(query)
    data = query_job.result().to_dataframe()
    return data

def generate_oil_leak_data(column, start_date, end_date, n_points=1, failure_mean=None, failure_std=None):
    failure_period_data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]

    mean = failure_period_data[column].mean()
    std = failure_period_data[column].std()

    # Set the mean and std for the failure period if provided
    if failure_mean is not None:
        mean = failure_mean
    if failure_std is not None:
        std = failure_std

    new_data = np.random.normal(mean, std, n_points)

    if column in ['comp', 'dv_electric', 'towers', 'mpg', 'lps', 'pressure_switch', 'dv_eletric', 'oil_level', 'caudal_impulses', 'gps_speed', 'gps_quality']:
        return np.round(new_data).astype(int)
    else:
        return new_data
def plot_data_from_bigquery(data):
    data.reset_index(inplace=True)

    with st.container():
        plot1, plot2, plot3, plot4 = st.columns(4)
        plot1.metric(label="TP3 ⏳", value=data['TP3'].iloc[-1], delta= data['TP3'].iloc[-1] - 10)
        plot2.metric(label="DV_pressure ⏳", value=data['DV_pressure'].iloc[-1], delta= data['DV_pressure'].iloc[-1] - 10)
        plot3.metric(label="Oil_temp ⏳", value=data['Oil_temperature'].iloc[-1], delta=data['Oil_temperature'].iloc[-1] - 10)

        fig_row1, fig_row2, fig_row3 = st.columns(3)
        with fig_row1:
            st.markdown("### TP3 stream")
            fig = go.FigureWidget(px.line(data_frame=data, y='TP3', x='timestamp'))
            st.write(fig)
        with fig_row2:
            st.markdown("### DV_pressure stream")
            fig2 = go.FigureWidget(px.line(data_frame=data, y='DV_pressure', x='timestamp'))
            st.write(fig2)
        with fig_row3:
            st.markdown("### Oil_temperature stream")
            fig3 = go.FigureWidget(px.line(data_frame=data, y='Oil_temperature', x='timestamp'))
            st.write(fig3)

    return fig, fig2, fig3

def send_to_pubsub(live_data_row):
    import json
    import base64

    data_str = live_data_row.to_json()
    data_bytes = data_str.encode("utf-8")
    data_b64 = base64.b64encode(data_bytes).decode('utf-8')

    message = {
        "data": data_b64
    }

    message_str = json.dumps(message)
    message_bytes = message_str.encode("utf-8")

    print(f"Sending message to Pub/Sub: {message_str}")  # Add this print statement

    future = publisher.publish(topic_path, data=message_bytes)
    return future
def segment_intervals(times, n_segments):
    return [np.linspace(0, interval, n_segments + 1, endpoint=True, dtype=int) for interval in times]

def compute_mean_and_multiply(data, intervals, cycle_duration):
    mean_values = []

    for interval in intervals:
        mean_interval = np.mean(data[interval[0]:interval[-1]])
        mean_values.append(mean_interval * cycle_duration)

    return mean_values
def display_prediction(prediction_array):
    st.write("Failure Prediction:")

    if np.all(prediction_array == 1):
        st.error("⚠️ Failure!")
    else:
        st.success("✅ Non-Failure.")


#@st.cache_resource 
def predict_failure(X_pred):
    X_pred_outputs = model.predict(X_pred)
    alpha=0.04
    threshold=2.0310641026292667e-19
    # Compute reconstruction error for X_pred
    er_pred = np.abs(X_pred - X_pred_outputs)

    # Apply the Low Pass Filter (LPF) to er_pred
    er_pred_filtered = alpha * er_pred[:-1] + (1 - alpha) * er_pred[1:]

    # Compute er_thresholding by comparing the filtered er_pred with the threshold
    er_pred_filtered = er_pred > threshold

    # Assign anomaly labels (1 for anomaly, 0 for normal) based on er_thresholding
    y_pred = er_pred_filtered.astype(int)
    return y_pred


def display_prediction(prediction_array, message_placeholder):
    num_ones = np.sum(prediction_array == 1)
    num_zeros = np.sum(prediction_array == 0)

    print(f"Number of ones: {num_ones}, Number of zeros: {num_zeros}")

    if num_ones > num_zeros:
        message_placeholder.error("⚠️ Failure!")
    elif num_ones <= num_zeros:
        message_placeholder.success("✅ Non-Failure.")


def simulate_data_to_pubsub():
    st.write("Simulate and send all sensor data to Pub/Sub")

    if "start_button" not in st.session_state:
        st.session_state.start_button = False

    if "stop_button" not in st.session_state:
        st.session_state.stop_button = False

    if st.button("Start Live Data"):
        st.session_state.start_button = True
        st.session_state.stop_button = False
        
    if st.button("Stop Live Data"):
        st.session_state.start_button = False
        st.session_state.stop_button = True

    if st.session_state.start_button and not st.session_state.stop_button:
        start_time = pd.Timestamp.now()
        chart_placeholder1 = st.empty()
        chart_placeholder2 = st.empty()
        chart_placeholder3 = st.empty()
        message_placeholder = st.empty()
        while not st.session_state.stop_button:
            new_row = {'timestamp': pd.Timestamp.now()}
            for col in data.columns:
                if col != 'timestamp':
                    new_row[col] = generate_live_data(col)[0]
            future = send_to_pubsub(pd.Series(new_row))
            print("Sent new data point to Pub/Sub:", new_row)

            updated_data = get_data_from_bigquery()
            updated_data.set_index('timestamp', inplace=True)
            data_copy = updated_data.copy()
            data_copy.reset_index(inplace=True)

            # Create new charts with updated data
            chart1 = px.line(data_frame=data_copy, y='TP3', x='timestamp')
            chart2 = px.line(data_frame=data_copy, y='DV_pressure', x='timestamp')
            chart3 = px.line(data_frame=data_copy, y='Oil_temperature', x='timestamp')
            
            # Update the existing charts
            chart_placeholder1.write(chart1)
            chart_placeholder2.write(chart2)
            chart_placeholder3.write(chart3)

            processed_data = preprocess_data(pd.DataFrame([new_row]))
            failure_prediction = predict_failure(processed_data)
            time.sleep(1)
            display_prediction(failure_prediction, message_placeholder)
        st.write("Live data streaming stopped.")

simulate_data_to_pubsub()

def simulate_oil_leak_data_to_pubsub():
    st.write("Simulate and send oil leak data to Pub/Sub")

    if "start_leak_button" not in st.session_state:
        st.session_state.start_leak_button = False

    if "stop_leak_button" not in st.session_state:
        st.session_state.stop_leak_button = False

    if st.button("Start Oil Leak Simulation"):
        st.session_state.start_leak_button = True
        st.session_state.stop_leak_button = False

    if st.button("Stop Oil Leak Data"):
        st.session_state.start_leak_button = False
        st.session_state.stop_leak_button = True

    if st.session_state.start_leak_button and not st.session_state.stop_leak_button:
        #Define the oil leak period or a longer period to get a mixture of normal and failing data
        start_date = '2022-03-23 14:54:00' 
        end_date = '2022-03-23 15:24:00'
        chart_placeholder1 = st.empty()
        chart_placeholder2 = st.empty()
        chart_placeholder3 = st.empty()

        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        historical_data_filtered = get_filtered_data_from_bigquery(start_date, end_date)

        print(f"Length of historical_data_filtered: {len(historical_data_filtered)}")
        message_placeholder = st.empty()
        for i, row in historical_data_filtered.iterrows():
            if st.session_state.stop_leak_button:
                st.write("Oil leak data streaming stopped.")
                break

            new_row = row.to_dict()

            future = send_to_pubsub(pd.Series(new_row))
            updated_data = get_data_from_bigquery()
            updated_data.set_index('timestamp', inplace=True)
            data_copy = updated_data.copy()
            data_copy.reset_index(inplace=True)
             # Create new charts with updated data
            chart1 = px.line(data_frame=data_copy, y='TP3', x='timestamp')
            chart2 = px.line(data_frame=data_copy, y='DV_pressure', x='timestamp')
            chart3 = px.line(data_frame=data_copy, y='Oil_temperature', x='timestamp')
            
            # Update the existing charts
            chart_placeholder1.write(chart1)
            chart_placeholder2.write(chart2)
            chart_placeholder3.write(chart3)
            
            processed_data = preprocess_data(pd.DataFrame([new_row]))
            failure_prediction = predict_failure(processed_data)
            time.sleep(1)
            
            display_prediction(failure_prediction, message_placeholder)
           

simulate_oil_leak_data_to_pubsub()


