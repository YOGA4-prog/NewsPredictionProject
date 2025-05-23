import pandas as pd

# Load datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels
fake['label'] = 'FAKE'
real['label'] = 'REAL'

# Combine datasets
data = pd.concat([fake, real], axis=0)
data = data.sample(frac=1).reset_index(drop=True)  # shuffle

# Check the data
print(data.shape)
print(data['label'].value_counts())
data.head()
