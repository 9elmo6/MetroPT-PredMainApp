from google.cloud import bigquery
import base64
import json
from datetime import datetime

def save_to_bigquery(data, context):
    # Configure the BigQuery client and the target dataset and table
    client = bigquery.Client()
    table_id = "metroPT-Pdm.apuHistoryData.apuDataTable" # change here
    raw_data = json.loads(base64.b64decode(data['data']).decode('utf-8'))

    # Decode the Pub/Sub message and parse it as JSON
    message = json.loads(base64.b64decode(raw_data['data']).decode('utf-8'))
    print(f"Decoded message: {message}")

    timestamp_str = datetime.utcfromtimestamp(message["timestamp"] / 1000).strftime('%Y-%m-%d %H:%M:%S.%f')
    print(f"Original timestamp: {message['timestamp']}")
    print(f"Converted timestamp: {timestamp_str}")

    print(f"Received message: {message}")

    # Build the row data to be inserted into BigQuery
    row_data = [
        {
                "timestamp": timestamp_str,
                "TP2": message["tp2"],
                "TP3": message["tp3"],
                "H1": message["h1"],
                "DV_pressure": message["dv_pressure"],
                "Reservoirs": message["reservoirs"],
                "Oil_temperature": message["oil_temperature"],
                "Flowmeter": message["flowmeter"],
                "Motor_current": message["motor_current"],
                "COMP": message["comp"],
                "DV_eletric": message["dv_eletric"],
                "Towers": message["towers"],
                "MPG": message["mpg"],
                "LPS": message["lps"],
                "Pressure_switch": message["pressure_switch"],
                "Oil_level": message["oil_level"],
                "Caudal_impulses": message["caudal_impulses"],
                "gpsLong": message["gpslong"],
                "gpsLat": message["gpslat"],
                "gpsSpeed": round(message["gpsspeed"]),
                "gpsQuality": round(message["gpsquality"]),
            }
    ]
    print("Rows to insert:", row_data)

    # Insert the data into the BigQuery table
    errors = client.insert_rows_json(table_id, row_data)
    if errors:
        raise RuntimeError(errors)
    else:
        print(f"Inserted 1 row into {table_id}")