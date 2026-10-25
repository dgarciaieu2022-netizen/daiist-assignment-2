"""
Every number the decision depends on, in one place.

train.py, app.py and REPORT.md all read from here, so a figure can never
disagree between the pipeline, the dashboard and the write-up. Each one is
argued for in REPORT.md ("Cost matrix and operating point").
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Data: the exact Roboflow Universe version this pipeline was built on ----
RF_WORKSPACE = "dilusha-nadeeshan"
RF_PROJECT = "egg_damge_non_damage"
RF_VERSION = 1
RF_FORMAT = "yolov11"

POSITIVE_CLASS = "cracked egg"     # what "positive" means everywhere below
NEGATIVE_CLASS = "Uncracked egg"

# --- Model and training settings (recorded in REPORT.md) ---------------------
BASE_MODEL = "yolo11s.pt"          # COCO-pretrained; also the zero-shot baseline
EPOCHS = 100
PATIENCE = 25                      # stop early if val mAP stops improving
IMGSZ = 640                        # the dataset is already 640x640
BATCH = 16
SEED = 0

# --- Paths --------------------------------------------------------------------
DATASET_DIR = ROOT / "datasets" / f"{RF_PROJECT}-{RF_VERSION}"
MODELS_CACHE = ROOT / ".models"    # where the pretrained checkpoint is downloaded
RUNS_DIR = ROOT / "runs"
WEIGHTS_PATH = ROOT / "weights" / "egg_crack_yolo11s.pt"
ARTIFACTS = ROOT / "artifacts"     # everything app.py needs, committed

# --- The business: a small packing station that hand-grades eggs -------------
# One "case" = one egg. The model gives each egg a score: the highest
# confidence of any "cracked egg" box it finds (0 if it finds none).
#   score <  T_LOW               -> PASS   (packed as class A)
#   T_LOW <= score < T_HIGH      -> REVIEW (a person looks at it)
#   score >= T_HIGH              -> FAIL   (diverted to the industrial channel)

BASE_RATE = 0.05          # share of eggs arriving at the station that are cracked

COST_FN = 3.00            # EUR: cracked egg passed and packed into a retail box
COST_FP = 0.10            # EUR: sound egg failed and sold as industrial egg
COST_TP = 0.00            # EUR: cracked egg correctly failed (the baseline outcome)
COST_TN = 0.00            # EUR: sound egg correctly passed

HOURLY_WAGE = 16.0        # EUR/h, fully loaded cost of a packing operator
SECONDS_PER_REVIEW = 3.0  # time for a person to look at one flagged egg
COST_REVIEW = HOURLY_WAGE * SECONDS_PER_REVIEW / 3600   # EUR per reviewed egg

HUMAN_RECALL = 0.95       # share of cracked eggs a person catches when looking
HUMAN_FALSE_FAIL = 0.01   # share of sound eggs a person wrongly rejects

EGGS_PER_DAY = 20_000     # roughly a 25,000-hen farm grading its own eggs
PRODUCTIVE_HOURS_PER_SHIFT = 7.0  # an 8 h shift minus breaks and changeovers
