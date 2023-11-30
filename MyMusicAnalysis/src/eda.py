import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def eda_summary(self):
        """
        Perform Exploratory Data Analysis (EDA) and provide a summary.

        Returns:
        - Summary statistics for each attribute
        - Graphical analysis using Matplotlib and Seaborn
        """
        # Write code to read the dataset
        data = pd.read_csv(self.dataset_path)

        # Display summary statistics
        print("Summary Statistics:")
        print(data.describe())

        # Generate graphical analysis using Matplotlib and Seaborn
        # Example: Histogram for the first column
        plt.figure(figsize=(8, 6))
        sns.histplot(data.iloc[:, 0], bins=20, kde=True)
        plt.title('Histogram of Attribute')
        plt.xlabel('Attribute Values')
        plt.ylabel('Frequency')
        
        # Save the plot as an image
        plt.savefig('histogram_plot.png')
        
        # Show the plot
        plt.show()

if __name__ == "__main__":
    # Example Usage
    dataset_path = "C:/Users/emman/OneDrive/Desktop/MyMusicAnalysis/Acoustic Features.csv"  # Update the path accordingly
    analysis = EDA(dataset_path)  # Fix the class name to EDA
    analysis.eda_summary()
