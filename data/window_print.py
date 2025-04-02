import json
import numpy as np
import os

file = "data/windows/raw/10_window_0.npz"
data = np.load(file, allow_pickle=True)

output = {
    'data': data['data'].tolist(),
    'features': data['features'].tolist() if 'features' in data else None
}

print(output)