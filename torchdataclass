import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class FuturesGenerativeDataset(Dataset):
    def __init__(self, data_path, window_size=24):
        self.window_size = window_size
        self.samples = []

        # Load CSV
        df = pd.read_csv(data_path, parse_dates=["Date"])
        df = df.sort_values("Date")

        # Features: OHLCV + log return -> Log Return not needed
        #df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
        #df = df.dropna()

        features = df[["Open", "High", "Low", "Close", "Volume"]].values

        # Build rolling windows
        for i in range(len(features) - window_size):
            window = features[i:i+window_size]
            self.samples.append(window)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        return torch.tensor(x, dtype=torch.float)


# Use dataset
#temp
dataset = FuturesGenerativeDataset("BZ.csv", window_size=60)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for X in dataloader:
    print(X.shape)  # torch.Size([32, 60, 6])
    break

# Save one batch to disk
torch.save(X, "features.pt")
