import sys
from pathlib import Path
import torch

MODEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_DIR))

from absa_v14.inference import ABSAEngine

device = "cuda" if torch.cuda.is_available() else "cpu"

engine = ABSAEngine.from_pretrained(
    MODEL_DIR,
    device=device,
)

text = (
    "Tempatnya nyaman dan kopinya enak, "
    "tetapi pelayanannya lama."
)

results = engine.predict(
    text,
    profile="maps_high_recall",
)

for result in results:
    print(result)
