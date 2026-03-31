import json
import logging
import pandas as pd
from pathlib import Path
from src.preprocessor import TitanicPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def main() -> None:
    with open('config.json') as f:
        config = json.load(f)
    logger.info('Config loaded')

    df = pd.read_csv(config['data_url'])
    logger.info(f"Data loaded: {df.shape}")

    split_index = int(len(df) * (1 - config['test_size']))
    train_df, test_df = df.iloc[:split_index], df.iloc[split_index:]

    preprocessor = TitanicPreprocessor()
    train_clean = preprocessor.fit_transform(train_df)
    test_clean = preprocessor.transform(test_df)
    logger.info(f"Data preprocessed: Train shape {train_clean.shape} | Test shape {test_clean.shape}")


    Path('outputs').mkdir(exist_ok=True)
    train_clean.to_csv('outputs/train_clean.csv', index=False)
    test_clean.to_csv('outputs/test_clean.csv', index=False)
    logger.info("Clean data saved to outputs/")

if __name__ == "__main__":
    main()

