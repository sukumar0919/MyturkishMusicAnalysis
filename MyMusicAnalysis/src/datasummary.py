# datasummary.py

# Import necessary libraries
import pandas as pd

# Read the dataset
dataset_path = "../Acoustic Features.csv"  # Adjust the path based on your actual structure
data = pd.read_csv(dataset_path)

# Display basic information about the dataset
print("Data Summary:")
print(data.info())

# Additional details about the dataset structure
# (Modify this based on the actual structure of your dataset)
num_use_cases = len(data)
num_attributes = len(data.columns)
attributes_data_types = {column: data[column].dtype for column in data.columns}

print("\nAdditional Details:")
print(f"Number of Use Cases: {num_use_cases}")
print(f"Number of Attributes: {num_attributes}")

print("\nAttributes and Data Types:")
for attribute, data_type in attributes_data_types.items():
    print(f"- {attribute}: {data_type}")
