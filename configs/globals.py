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
    DATASET_PATH = "/content/drive/MyDrive/Master Project/dataset_axel/datasets/3d/CHUV/dataset_snapshot/"
else:
    raise ValueError("Unknown directory structure")

DATASET_PATH = os.path.abspath(DATASET_PATH)
OUTPUTS_PATH = os.path.abspath(OUTPUTS_PATH)

if debug:
    DATASET_PATH = os.path.join(DATASET_PATH, "suadd-test")
else:
    DATASET_PATH = os.path.join(DATASET_PATH, "suadd")

CLASSES = {i: c for i, c in enumerate([
    'water', 'asphalt', 'grass', 'human',
    'animal', 'high_vegetation', 'ground_vehicle', 'façade',
    'wire', 'garden_furniture', 'concrete', 'roof',
    'gravel', 'soil', 'primeair_pattern', 'snow', 'unknown'])}

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(OUTPUTS_PATH, exist_ok=True)
