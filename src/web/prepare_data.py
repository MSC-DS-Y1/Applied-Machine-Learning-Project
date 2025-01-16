from sklearn import preprocessing
import pandas as pd
from sklearn.decomposition import PCA

class DataPreparation:
    def __init__(self, inputs: pd.DataFrame):
        # Load the dataset
        self.dataset = pd.read_csv("../dataset/dataset_predictors.csv")
        
        # Separate predictors (features) and target
        self.predictors = self.dataset.drop('target', axis=1)
        self.target = self.dataset['target']
        
        # Save input data for prediction (ensure it aligns with the original predictors)
        self.inputs = inputs.reindex(columns=self.predictors.columns, fill_value=0)
        self.__pca_components = 14

    def apply_pca(self):
      # Normalize the features using StandardScaler
      scaler = preprocessing.StandardScaler()

      # Fit scaler on the training data predictors
      scaler.fit(self.predictors)

      # Transform the input data using the fitted scaler
      # Convert standardized data to DataFrame with original feature names
      
      X_std_predictors = scaler.transform(self.predictors)
      X_std_predictors_df = pd.DataFrame(X_std_predictors, columns=self.predictors.columns)
      
      X_std = scaler.transform(self.inputs)
      X_std_df = pd.DataFrame(X_std, columns=self.inputs.columns)

      # Apply PCA
      pca = PCA(n_components=self.__pca_components)
      pca.fit(X_std_predictors_df)
      X_pca = pca.transform(X_std_df)  # Transform the standardized inputs

      # Create a DataFrame for PCA-transformed features
      df_pca_transformed = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(self.__pca_components)])
      
      return df_pca_transformed