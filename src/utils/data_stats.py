import pandas as pd
from src.data.emotions_dict import EMOTION_DICT


"""Utils for analyze data
    1. class_distribution
"""

def get_class_distribution(csv_path: str) -> pd.Series:
    """
    Get class distribution for dataset

    Args:
        csv_path   : đường dẫn tới file CSV (có cột 'emotion')
    Return: 
        counts -> Series
    """
    df = pd.read_csv(csv_path)

    if 'emotion' not in df.columns:
        raise ValueError("CSV must contain an 'emotion' column")

    counts = (
        df['emotion']
        .value_counts()
        .reindex(EMOTION_DICT.keys(), fill_value=0)
        .astype(int)
    )

    return counts



# Unit test:
if __name__ == "__main__":
    from pathlib import Path
    from os.path import join
    cwd = Path.cwd().resolve()
    train_path = join(cwd, "dataset/fer13-split/train.csv")
    test_path = join(cwd, "dataset/fer13-split/test.csv")

    train_class_counts = get_class_distribution(train_path)
    test_class_counts = get_class_distribution(test_path)

    print("Train:", train_class_counts)
    print("Test:", test_class_counts)
    # print("1:disgust:",train_class_counts[1])
    # print(type(train_class_counts))


