import os

# Define the root folder name
root_folder = "PyTorch-Intermediate"

# Define subfolders based on the CampusX PyTorch playlist
subfolders = [
    "01_Tensors",
    "02_Autograd",
    "03_Training_Pipeline",
    "04_NN_Module",
    "05_Dataset_DataLoader",
    "06_ANN_Building",
    "07_GPU_Training",
    "08_NN_Optimization",
    "09_Hyperparameter_Tuning",
    "10_CNN_Building",
    "11_Transfer_Learning",
    "12_RNN_QA_System",
    "13_LSTM_Next_Word_Prediction"
]

# Create the root folder if it doesn't exist
if not os.path.exists(root_folder):
    os.makedirs(root_folder)

# Create subfolders inside the root folder
for subfolder in subfolders:
    path = os.path.join(root_folder, subfolder)
    os.makedirs(path, exist_ok=True)

print(f"Folder structure created under '{root_folder}' successfully!")
