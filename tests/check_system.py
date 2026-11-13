"""Checks that the whole project works. Run from the project root:  python tests/check_system.py"""
import contextlib
import io
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import cv2
import pandas as pd
from skimage import data as sample_images

from src.cli import main as cli
from src.config import AUDIO_CSV, FACE_MODEL, MERGED_CSV, PRODUCT_MODEL, VOICE_MODEL
from src.recommendation.product_model import recommend
from src.voice_verification.voice_model import FEATURE_COLUMNS, clean_name, extract_features, verify_voice

IMAGE_CSV = ROOT / "image_processing" / "data" / "image_features.csv"
LINDA_VOICE = ROOT / "audio_data" / "Linda_yes_approve.wav"
ARNOLD_VOICE = ROOT / "audio_data" / "converted" / "arnold_yes_approve.wav"

failures = []


def check(name, passed):
    print(("PASS  " if passed else "FAIL  ") + name)
    if not passed:
        failures.append(name)


def run_quietly(photo, audio):
    """Run the CLI without printing its messages, and return True/False for access."""
    with contextlib.redirect_stdout(io.StringIO()):
        return cli.run_access(str(photo), str(audio))


# 1. Files and datasets
for path in [MERGED_CSV, AUDIO_CSV, IMAGE_CSV, FACE_MODEL, PRODUCT_MODEL, VOICE_MODEL]:
    check(f"file exists: {path.relative_to(ROOT)}", path.exists())

merged = pd.read_csv(MERGED_CSV)
check("merged dataset: no missing values", merged.isna().sum().sum() == 0)
check("merged dataset: no duplicate transactions", not merged["transaction_id"].duplicated().any())

images = pd.read_csv(IMAGE_CSV)
check("image features: every member has 3 expressions", (images.groupby("member_id")["expression"].nunique() == 3).all())
check("image features: at least 2 augmentations", images["augmentation"].nunique() >= 2)

audio = pd.read_csv(AUDIO_CSV)
check("audio features: columns match the voice model", list(audio.columns[4:]) == FEATURE_COLUMNS)
audio["member"] = audio["member_name"].apply(clean_name)
check("audio features: 4 members x 2 phrases", audio["member"].nunique() == 4 and (audio.groupby("member")["phrase"].nunique() == 2).all())
check("audio and face models use the same member names", set(audio["member"]) == set(images["member_id"]))

# 2. Voice model
row = audio[(audio["filename"] == LINDA_VOICE.name) & (audio["augmentation"] == "none")][FEATURE_COLUMNS].iloc[0]
new = pd.Series(extract_features(LINDA_VOICE))[FEATURE_COLUMNS]
check("voice features from a wav file match audio_features.csv", ((new - row).abs() / (row.abs() + 1e-9)).max() < 1e-3)
check("voice check accepts the right person", verify_voice(LINDA_VOICE, "linda")[0])
check("voice check rejects the wrong person", not verify_voice(LINDA_VOICE, "arnold")[0])

# 3. Recommendation model
suggestions = recommend(100)
check("recommendation returns 3 categories", len(suggestions) == 3)
check("recommendation probabilities are valid", all(0 <= p <= 1 for _, p in suggestions) and sum(p for _, p in suggestions) <= 1.0001)

# 4. Command-line app
# A stranger photo that ships with scikit-image (real face model, no private photos needed)
stranger = Path(tempfile.gettempdir()) / "stranger_check.png"
cv2.imwrite(str(stranger), cv2.cvtColor(sample_images.astronaut(), cv2.COLOR_RGB2BGR))
check("CLI denies a stranger's face (real face model)", not run_quietly(stranger, LINDA_VOICE))
check("CLI denies a missing photo", not run_quietly(ROOT / "no_such_photo.jpg", LINDA_VOICE))

# The project has no member photos (privacy), so the face result is replaced by a fixed answer
# to test the voice and recommendation steps. This tests the wiring, not the face model.
cli.predict_face = lambda *args, **kwargs: ("linda", 0.9)
check("CLI grants access when face and voice match (face result simulated)", run_quietly(stranger, LINDA_VOICE))
check("CLI denies a voice from a different person (face result simulated)", not run_quietly(stranger, ARNOLD_VOICE))
check("CLI denies a missing audio file (face result simulated)", not run_quietly(stranger, ROOT / "no_such_voice.wav"))

print()
if failures:
    print(f"{len(failures)} check(s) FAILED. Fix them before committing.")
    sys.exit(1)
(ROOT / ".checks_passed").write_text("ok")
print("ALL CHECKS PASSED")
