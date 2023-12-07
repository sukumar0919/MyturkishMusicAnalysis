# datasummary.py

# Import necessary libraries
import pandas as pd

def display_data_summary(dataset_path):
    """
    Display a summary of the dataset, including basic information, additional details, and attributes' data types.

    Parameters:
    - dataset_path (str): Path to the dataset CSV file.

    Returns:
    - None
    """
    # Read the dataset
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

# Example Usage
if __name__ == "__main__":
    dataset_path = "../Acoustic Features.csv"  # Adjust the path based on your actual structure
    display_data_summary(dataset_path)
