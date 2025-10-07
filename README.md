Predicting Metabolic Cost from Wearable Sensor Data
Project for CS229: Machine Learning

1. Project Overview
This project tackles the challenge of predicting the metabolic cost of human locomotion using data from wearable sensors. The goal is to develop a machine learning model that can accurately estimate metabolic rate from gait kinematics and muscle activity (EMG), which has significant applications in exoskeleton control, rehabilitation, and assistive device optimization.

We compare the performance of a regularized linear model (Lasso) against a non-linear model (a single-layer Neural Network) to determine the most effective approach for this complex regression task. The pipeline involves extracting data from a complex .mat file, generating synthetic features, preprocessing the data, and training/evaluating the models using k-fold cross-validation.

The final results demonstrate that a Neural Network significantly outperforms the linear model, indicating the presence of complex, non-linear relationships in the data.

2. Repository Contents
This repository contains the full source code required to reproduce the project's findings.

generate_data.py: A robust Python script that parses the complex P01.mat file, extracts all EMG data chunks, generates synthetic features, and creates the final realistic_synthetic_data.csv dataset.

data_processing.py: A script that takes the generated dataset, performs Z-score normalization, and saves the preprocessed data.

model_train.py: The final script in the pipeline that loads preprocessed data, trains all models, and outputs the final performance metrics to results.json.

P01.mat: (Not included in repository) The raw sensor data file. Due to its large size, this file must be downloaded from the original source (see Setup instructions).

3. Setup and Execution Instructions
Follow these steps to set up the environment and run the complete project pipeline.

Step 1: Clone the Repository
Clone this private GitHub repository to your local machine.

Step 2: Download the Raw Data
The raw .mat data files are hosted publicly on Figshare. You must download the dataset before running the scripts.

Go to the data source: https://figshare.com/articles/dataset/High-density_EMG_IMU_Kinetic_and_Kinematic_Open-Source_Dataset/22227337?file=42558349

Download the P01.zip file (approximately 61 MB).

Unzip the file. Inside, you will find the P01.mat file.

Place the P01.mat file inside the data/ directory of this project.

Step 3: Set Up the Python Environment
It is highly recommended to use a virtual environment.

# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Step 4: Install Required Libraries
This project requires several scientific computing libraries. The following command installs versions that are known to be compatible and avoid known version conflicts.

pip install "numpy<2.0" scipy pandas scikit-learn matplotlib

Step 5: Run the Full Pipeline
Execute the three main Python scripts in the correct order from your terminal.

1. Generate the Dataset:
This script reads P01.mat and creates the realistic_synthetic_data.csv file. It will provide progress updates as it processes chunks of data.

python generate_data.py

2. Preprocess the Data:
This normalizes the dataset. This step is very fast.

python data_processing.py realistic_synthetic_data.csv --out preprocessed.csv

3. Train the Models:
This runs all model training and produces the final results.json and results_performance.png files.

python model_train.py preprocessed.csv --out results.json

After these commands complete, the results.json file will contain the final performance metrics for each model, summarizing the project's findings.
