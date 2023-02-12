import os

debug = True

# For my own computer
if os.getcwd().split(os.path.sep)[0] == "C:":
    DATASET_PATH = "C:/Users/Axeld/Desktop/AICrowd/suadd/datasets/"
    OUTPUTS_PATH = "C:/Users/Axeld/Desktop/AICrowd/suadd/outputs/"
# For clusters
elif os.getcwd().split(os.path.sep)[1] == "home":
    DATASET_PATH = "/home/dinhvan/master_project/dataset_axel/datasets/3d/CHUV/dataset_snapshot"
# For Google Colab
elif os.getcwd().split(os.path.sep)[1] == "content":
    DATASET_PATH = "/content/drive/MyDrive/AIcrowd/suadd/suadd/datasets"
    OUTPUTS_PATH = "/content/drive/MyDrive/AIcrowd/suadd/suadd/outputs"
else:
    raise ValueError("Unknown directory structure")

DATASET_PATH = os.path.abspath(DATASET_PATH)
OUTPUTS_PATH = os.path.abspath(OUTPUTS_PATH)

if debug:
    DATASET_PATH = os.path.join(DATASET_PATH, "suadd-test")
else:
    DATASET_PATH = os.path.join(DATASET_PATH, "suadd")

CLASSES = {
    0: "water",
    1: "asphalt",
    2: "grass",
    3: "human",
    4: "animal",
    5: "high_vegetation",
    6: "ground_vehicle",
    7: "façade",
    8: "wire",
    9: "garden_furniture",
    10: "concrete",
    11: "roof",
    12: "gravel",
    13: "soil",
    14: "primeair_pattern",
    15: "snow",
    16: "unknown",
}

PALETTE = [
    (148, 218, 255),  # light blue, WATER
    (85, 85, 85),  # almost black, ASPHALT
    (200, 219, 190),  # light green, GRASS
    (166, 133, 226),  # purple, HUMAN
    (255, 171, 225),  # pink, ANIMAL
    (40, 150, 114),  # green, HIGH VEGETATION
    (234, 144, 133),  # orange, GROUND VEHICLE
    (89, 82, 96),  # dark gray, FACADE
    (255, 255, 0),  # yellow, WIRE
    (110, 87, 121),  # dark purple, GARDEN FURNITURE
    (205, 201, 195),  # light gray, CONCRETE
    (212, 80, 121),  # medium red, ROOF
    (159, 135, 114),  # light brown, GRAVEL
    (102, 90, 72),  # dark brown, SOIL
    (255, 255, 102),  # bright yellow, PRIMEAIR PATTERN
    (251, 247, 240),  # almost white, SNOW
    (0, 0, 0),  # black, UNKNOWN
]

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(OUTPUTS_PATH, exist_ok=True)
