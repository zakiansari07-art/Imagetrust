from pathlib import Path
import pickle

history_path = "src/training/history/history.pkl"
history = [1,3,4]
with open(history_path, "wb") as f:
    pickle.dump(history, f)