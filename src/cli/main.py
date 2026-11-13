"""Command-line app: face check -> voice check -> product recommendation.

Interactive:   python -m src.cli.main
With files:    python -m src.cli.main <face_photo> <voice_recording>
"""
import sys
from pathlib import Path

from src.config import FACE_MODEL, FACE_SCRIPT_DIR, PRODUCT_MODEL, VOICE_MODEL
from src.recommendation.product_model import recommend
from src.voice_verification.voice_model import verify_voice

sys.path.insert(0, str(FACE_SCRIPT_DIR))
from face_auth import predict_face  # the facial recognition model built by the team

# DEMO ONLY: team members are not customers in the dataset, so each one is given a sample customer
DEMO_CUSTOMERS = {"arnold": 100, "crispin": 101, "gaju": 102, "linda": 103}


def ask_for_file(question):
    return input(question).strip().strip("\"'")   # removes quotes added by drag-and-drop


def check_face(photo_path):
    """Return the recognised member's name, or None if access should be denied."""
    if not Path(photo_path).is_file():
        print(f"  File not found: {photo_path}")
        return None
    try:
        member, confidence = predict_face(photo_path, model_path=FACE_MODEL, details=True)
    except Exception as error:
        print(f"  Could not read the photo: {error}")
        return None

    if member == "unknown":
        print("  Face NOT recognised.")
        return None
    print(f"  Face matched: {member} (confidence {confidence:.2f})")
    return member


def check_voice(audio_path, member):
    """Return True only if the voice belongs to the same person the face check found."""
    if not Path(audio_path).is_file():
        print(f"  File not found: {audio_path}")
        return False
    try:
        verified, heard, confidence = verify_voice(audio_path, member)
    except Exception as error:
        print(f"  Could not read the recording: {error}")
        return False

    if not verified:
        print(f"  Voice does NOT match {member} (it sounds like {heard}, confidence {confidence:.2f}).")
        return False
    print(f"  Voice matched: {member} (confidence {confidence:.2f})")
    return True


def show_recommendation(member, recommendations):
    print(f"  Demo customer for {member}: {DEMO_CUSTOMERS[member]}")
    for rank, (category, probability) in enumerate(recommendations, start=1):
        print(f"  {rank}. {category} ({probability:.0%})")


def run_access(photo_path=None, audio_path=None):
    """Run the whole sequence. Asks for any file that was not given. Returns True if access is granted."""
    print("\nStep 1/3 - Facial recognition")
    member = check_face(photo_path or ask_for_file("  Path to your face photo: "))
    if member is None:
        print("\nACCESS DENIED (face check failed)\n")
        return False

    # Face recognised: the product model runs now, but the answer stays hidden until the voice is verified
    recommendations = recommend(DEMO_CUSTOMERS[member])
    print("  Product model has run. The result is hidden until the voice is verified.")

    print("\nStep 2/3 - Voice verification")
    if not check_voice(audio_path or ask_for_file("  Path to your voice recording: "), member):
        print("\nACCESS DENIED (voice check failed)\n")
        return False

    print("\nStep 3/3 - Product recommendation")
    show_recommendation(member, recommendations)
    print("\nACCESS GRANTED\n")
    return True


if __name__ == "__main__":
    if not (PRODUCT_MODEL.exists() and VOICE_MODEL.exists()):
        sys.exit("Models not found. Train them first:\n"
                 "  python -m src.recommendation.product_model\n"
                 "  python -m src.voice_verification.voice_model")
    args = sys.argv[1:]
    run_access(*args) if len(args) == 2 else run_access()
