import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

class Inference:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def perform_inference(self):
        # Load the dataset
        data = pd.read_csv(self.dataset_path)

        # Step 1: Select Relevant Acoustic Features
        selected_features = ['_Tempo_Mean', '_RMSenergy_Mean', '_Spectralcentroid_Mean', '_Fluctuation_Mean']
        emotions_column = 'Class'  # Replace with the actual column name for emotions

        # Step 1a: Encode Emotions into Numerical Values
        label_encoder = LabelEncoder()
        data['Encoded_Emotions'] = label_encoder.fit_transform(data[emotions_column])

        # Step 2: Explore Correlation with Emotions
        correlation_matrix = data[selected_features + ['Encoded_Emotions']].corr()

        # Step 3: Visualization
        plt.figure(figsize=(14, 6))

        # Matplotlib
        plt.subplot(1, 2, 1)
        plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none', aspect='auto')
        plt.title('Correlation Matrix (Matplotlib)')
        plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation='vertical')
        plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
        plt.colorbar()

        # Seaborn
        plt.subplot(1, 2, 2)
        sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True)
        plt.title('Correlation Matrix (Seaborn)')

        plt.tight_layout()
        plt.show()

        # Step 4: Narrative
        """
        Research Approach and Findings:

        - Selected key acoustic features: Tempo, RMS energy, Spectral centroid, and Fluctuation.
        - Encoded emotions into numerical values.
        - Explored correlations between these features and various emotions using a correlation matrix.
        - Visualizations using Matplotlib and Seaborn showcase the strength of correlations.

        Key Findings:
        - Positive correlations between specific acoustic features and emotions may indicate meaningful relationships.
        - Considerable variation in correlations emphasizes the complexity of the acoustic-emotion interplay in Turkish music.

        Further Steps:
        - Dive deeper into specific feature-emotion relationships for a more nuanced understanding.
        - Explore additional visualizations or statistical tests for a comprehensive analysis.

        This analysis provides an initial insight into the intricate connection between acoustic features and emotions in Turkish music.
        """

if __name__ == "__main__":
    # Example Usage
    dataset_path = "C:/Users/emman/OneDrive/Desktop/MyMusicAnalysis/Acoustic Features.csv"
    analysis = Inference(dataset_path)
    analysis.perform_inference()
