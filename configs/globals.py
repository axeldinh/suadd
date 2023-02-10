import os

debug = True

# For my own computer
if os.getcwd().split(os.path.sep)[0] == "C:":
    dataset_path = "C:/Users/Axeld/Desktop/AICrowd/suadd/datasets/"
    outputs_path = "C:/Users/Axeld/Desktop/AICrowd/suadd/outputs/"
# For clusters
elif os.getcwd().split(os.path.sep)[1] == "home":
    dataset_path = "/home/dinhvan/master_project/dataset_axel/datasets/3d/CHUV/dataset_snapshot"
# For Google Colab
elif os.getcwd().split(os.path.sep)[1] == "content":
    dataset_path = "/content/drive/MyDrive/Master Project/dataset_axel/datasets/3d/CHUV/dataset_snapshot/"
else:
    raise ValueError("Unknown directory structure")

dataset_path = os.path.abspath(dataset_path)

if debug:
    dataset_path = os.path.join(dataset_path, "suadd-test")
else:
    dataset_path = os.path.join(dataset_path, "suadd")
