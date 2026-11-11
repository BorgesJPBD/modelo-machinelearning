import os
import kagglehub
from kagglehub import KaggleDatasetAdapter

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "natezhang123/social-anxiety-dataset",
    "enhanced_anxiety_dataset.csv",
)

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/enhanced_anxiety_dataset.csv", index=False)
print(df.shape)  # deve mostrar (11000, 19)
