"""File locations, all relative to the project root so the code runs on any computer."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MERGED_CSV = ROOT / "data" / "merged_dataset.csv"
AUDIO_CSV = ROOT / "audio_features.csv"

FACE_SCRIPT_DIR = ROOT / "image_processing" / "scripts"   # where face_auth.py lives
FACE_MODEL = ROOT / "image_processing" / "models" / "face_model.pkl"
PRODUCT_MODEL = ROOT / "models" / "product_recommendation" / "product_model.pkl"
VOICE_MODEL = ROOT / "models" / "voice_verification" / "voice_model.pkl"
