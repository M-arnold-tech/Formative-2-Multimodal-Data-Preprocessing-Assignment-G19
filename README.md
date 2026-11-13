# Formative 2 - Multimodal Data Preprocessing (Group 19)

A command-line product recommendation system that only gives a recommendation after two checks pass:
**facial recognition** and **voiceprint verification**. This repository contains the data merge, the image and
audio feature extraction, the three models (face, voice, product) and the command-line app.

## How the system works

```
Face photo -> Face recognition -> recognised? -- no --> ACCESS DENIED
                                      | yes
                       Product model runs (result stays hidden)
                                      |
Voice recording -> Voice verification -> same person as the face? -- no --> ACCESS DENIED
                                      | yes
                       Product recommendation is displayed
```

The two checks must agree: the voice has to be identified as the same team member the face check found.
Team members are not customers in the dataset, so for the demo each member is given a **sample customer**
(arnold = 100, crispin = 101, gaju = 102, linda = 103). This is demo data only.

## Repository layout

```
data/                              customer CSVs, merged_dataset.csv (+ readable version)
scripts/data_preprocessing.py      builds merged_dataset.csv from the two raw CSVs
notebooks/eda_data_merge.ipynb     Part 1: cleaning, merge, feature engineering, EDA
image_processing/                  Part 2: image notebook (Colab), image_features.csv, face model, face_auth.py,
                                   train_face_model.py
Sound_Data_Collection_and_processing.ipynb   Part 3: audio notebook
audio_data/, audio_features.csv    recordings and their extracted features
src/recommendation/product_model.py          product model: train, evaluate, recommend
src/voice_verification/voice_model.py        voice model: features, train, evaluate, verify
src/cli/main.py                    the command-line app
src/config.py                      all file paths in one place
models/                            trained product and voice models + metrics.json
tests/check_system.py              checks that everything works
```

## Setup

Tested with Python 3.12.

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`scikit-learn` is pinned to 1.6.1 because the face model was saved with that version.

## Run

Rebuild the merged dataset (optional, the result is already in `data/`):

```bash
python scripts/data_preprocessing.py
```

Train and evaluate the two models built for the app (this also saves them in `models/`):

```bash
python -m src.recommendation.product_model
python -m src.voice_verification.voice_model
```

Retrain the face model from `image_processing/data/image_features.csv` (keeps the saved confidence threshold and distance limit):

```bash
python image_processing/scripts/train_face_model.py
```

Check that everything works:

```bash
python tests/check_system.py
```

Start the app (it asks for a face photo, then a voice recording):

```bash
python -m src.cli.main
```

or give both files directly:

```bash
python -m src.cli.main <face_photo> <voice_recording>
```

The notebooks are opened with `jupyter notebook`. The image notebook was written for Google Colab
(it reads the photos from Google Drive), so run it there.

Use `.wav` files for voice recordings. Other formats (such as `.m4a`) may need `ffmpeg` installed.

## Demonstration

Face photos are not in this repository. Put your own photos in `data/demo_photos/`
(this folder is ignored by Git), then run:

```bash
# 1. Successful access: a member's photo with the same member's voice
python -m src.cli.main data/demo_photos/linda.jpg audio_data/Linda_yes_approve.wav

# 2. Unauthorized attempt: a face that is not a team member
python -m src.cli.main data/demo_photos/stranger.jpg audio_data/Linda_yes_approve.wav

# 3. Unauthorized attempt: a member's face with a different person's voice
python -m src.cli.main data/demo_photos/linda.jpg audio_data/converted/arnold_yes_approve.wav
```

The demo recordings were also used to train the voice model, so the demo shows the flow of the system.
The model's accuracy is reported below from a separate test.

## Results

All numbers come from the code in this repository.

| Model | How it was tested | Accuracy | F1 | Log loss |
|---|---|---|---|---|
| Face (Logistic Regression) | train on two expressions, test on the third (12 photos) | 0.917 | 0.889 | 0.117 |
| Voice (Logistic Regression) | train on one phrase, test on the other (8 clips) | 1.000 | 1.000 | 0.186 |
| Product (Random Forest) | 5 test runs, a customer's rows never split between train and test | 0.185 | 0.167 | 1.735 |
| Product baseline (always the most common category) | same as above | 0.183 | 0.100 | 1.642 |

- **Face:** from `image_processing/MLP_Formative2_Imageprocessing.ipynb`. The saved model also rejects unknown faces
  with a confidence threshold of 0.67 and a distance limit of 32.1, tuned on only 2 outsider photos.
- **Voice:** all 8 held-out clips were identified correctly, but 8 clips is a very small test, so treat 1.000 as
  encouraging and not proven. The model rejects a team member's voice used under another member's name, but it
  **cannot reliably reject a stranger's voice**: when one speaker is hidden from the model, its confidence on that
  unknown voice (0.46 to 0.92) overlaps with its confidence on real members (0.64 to 0.97). Every recording is
  resampled to 22.05 kHz before the features are computed, so phone recordings at 44.1 or 48 kHz work. In one test,
  a recording limited to 16 kHz (nothing above 8 kHz) was still misidentified.
- **Product:** the model is no better than guessing. With 5 categories, chance is 0.20, and all three models score about
  0.18 to 0.19. The EDA already showed that the categories overlap strongly. Random Forest was chosen because it
  has the lowest log loss of the two real models, but its log loss is still worse than the baseline's.

Full numbers and confusion matrices are in `models/*/metrics.json`.

## Data

`data/merged_dataset.csv` has 150 transactions from 75 customers, 36 columns, no missing values and no duplicate
transactions. The target is `product_category` (Sports 35, Electronics 34, Clothing 28, Groceries 28, Books 25).
Image features (`image_processing/data/image_features.csv`) have 60 rows (4 members x 3 expressions x 5 versions)
with 516 features. Audio features (`audio_features.csv`) have 32 rows (4 members x 2 phrases x 4 versions) with 30 features.

The voice model does not read `audio_features.csv`. It computes the same 30 features (with the notebook's 3 augmentations)
from the recordings in `audio_data/` at a fixed 22.05 kHz, because the CSV was extracted at each file's own sample rate.
The audio files and notebook call one member "Keane" and the face model calls the same person "gaju"; the code maps `keane` to `gaju`.

## Team

Contribution sheet: 
