import pandas as pd

# Loads and preprocesses the Titanic dataset

class TitanicLoader:
    def __init__(self, url: str):
        self.url = url
        self.df = None

    def load(self) -> 'TitanicLoader':
        # Load the dataset from the specified file path
        self.df = pd.read_csv(self.file_path)
        print(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
        return self

    def info(self) -> None:
        # Print basic dataset information
        print(f"Shape: {self.df.shape}")
        print(f"Nulls:\n{self.df.isnull().sum()}")

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
loader = TitanicLoader(URL)
loader.load().info()

