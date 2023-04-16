import csv
import json

# Open the CSV file for reading
with open('dataset1000.csv', newline='') as csvfile:
    # Read in the CSV data
    reader = csv.DictReader(csvfile)

    # Create an empty list to hold the JSON data
    data = []

    # Loop over each row in the CSV file and convert it to JSON
    for row in reader:
        # Convert the row to a dictionary
        d = dict(row)

        # Append the dictionary to the data list
        data.append(d)

# Write the JSON data to a file
with open('output.json', 'w') as jsonfile:
    # Convert the data list to a JSON string
    json_string = json.dumps(data, indent=4)

    # Write the JSON string to the output file
    jsonfile.write(json_string)
