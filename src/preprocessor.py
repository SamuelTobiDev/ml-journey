import pandas as pd

from src.utils import load_data

class TitanicPreprocessor:
      # Clean and engineer features. Follows scikit-learn's fit-transform pattern.

    def __init__(self):
        self.age_median = None
        self.fare_median = None
        self.embarked_mode = None
        self.is_fitted = False 

    def fit(self, df: pd.DataFrame) -> 'TitanicPreprocessor':
        # Compute statistics for imputation and encoding
        self.age_median = df['Age'].median()
        self.fare_median = df['Fare'].median()
        self.embarked_mode = df['Embarked'].mode()[0]
        self.is_fitted = True
        print(f"Fitted Age median={self.age_median}")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Apply transformations using the fitted statistics
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before calling transform.")
        
        df = df.copy()
        # Impute missing values
        df['Age'] = df['Age'].fillna(self.age_median)
        df['Fare'] = df['Fare'].fillna(self.fare_median)
        df['Embarked'] = df['Embarked'].fillna(self.embarked_mode)
        df = df.drop(columns=['Cabin','Name','Ticket'])  # Drop less useful features
        df['Sex'] = df['Sex'].map({'male': 0,'female': 1})
        df['Embarked'] = df['Embarked'].map({'S': 0,'C': 1,'Q': 2})
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # Create new feature
        df['isAlone'] = (df['FamilySize'] == 1).astype(int)  # Create new feature
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
         # Convenience method to fit and transform in one step
        return self.fit(df).transform(df)   