# Histopathologic Cancer Detection

This repository contains a project for detecting metastatic cancer in histopathologic images of lymph node sections. The task is part of a Kaggle competition where the goal is to build a binary classifier to identify whether a small 96x96 pixel pathology image contains tumor tissue. 

## Project Overview

In this project, two models are explored:
1. A **Simple CNN** model, designed to balance simplicity with performance.
2. **ResNet50**, a pre-trained deep convolutional neural network, fine-tuned for this binary classification task.

The models are trained and evaluated on the PatchCamelyon dataset, with the goal of maximizing the area under the ROC curve (AUC) for robust cancer detection.

### Dataset
The dataset used is the modified version of the **PatchCamelyon (PCam)** dataset, containing 96x96 pixel images labeled with binary values (0 or 1). A positive label (1) indicates that the image contains tumor tissue.

## Project Structure

- `notebooks/`: Contains Jupyter notebooks for training and evaluating each model.
- `src/`: Holds the main Python scripts for data processing, model training, and evaluation.
- `README.md`: This file, explaining the project's goals, structure, results, and future improvements.
- `requirements.txt`: List of required packages to run the project.

## Installation

Clone this repository:
git clone https://github.com/your-username/histopathologic-cancer-detection.git
cd histopathologic-cancer-detection


Usage
Prepare Data: Download the dataset from Kaggle and place it in the data/ directory. The directory structure should look like this:

bash
Copy code
data/
├── train/
├── test/
└── train_labels.csv
Run Training: Use the notebooks in the notebooks/ directory to train the models. Each notebook contains instructions and configurations for training Simple CNN and ResNet50.

Evaluate Models: Each notebook also contains code for evaluating models and generating predictions. This can be modified for custom evaluations.

Make Predictions: Use the best trained model to predict on the test set and create a submission file for Kaggle.

Results
The models achieved the following validation performance:

Simple CNN: Achieved a validation AUC of ~0.93–0.94. Despite good convergence, the Kaggle leaderboard score was 0.7357, indicating challenges in generalizing to the test set.
ResNet50: Fine-tuning was more challenging, with validation AUC reaching ~0.89. The model struggled to improve further, likely due to limited adaptation with conservative learning rate adjustments.
Key Findings
Learning Rate Warm-Up: Gradually increasing the learning rate during the initial epochs stabilized training and improved convergence for both models.
Validation AUC Monitoring: Using validation AUC as a stopping criterion helped avoid overfitting, especially for the Simple CNN model.
Model Complexity: ResNet50, while more powerful, required careful tuning to avoid plateauing. The Simple CNN achieved better validation stability, suggesting that model complexity needs to be balanced with dataset characteristics.
Future Work
To enhance the models' performance and generalization, the following improvements are recommended:

Extended Fine-Tuning for ResNet50: Unfreeze additional layers and use a longer warm-up period with a higher initial learning rate after warm-up.
Cross-Validation: Implement cross-validation for more robust validation performance estimates, particularly for the complex ResNet50 model.
Additional Data Augmentation: Include augmentations such as color and brightness adjustments to increase robustness to varied images.
Alternative Architectures: Explore architectures optimized for transfer learning, such as EfficientNet or MobileNet.
Learnings
This project provided insights into the challenges of fine-tuning deep neural networks on medical imaging data. The importance of model selection, hyperparameter tuning, and validation strategies were underscored, particularly in achieving balanced performance on both validation and test data.

Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss potential improvements or issues.

License
This project is licensed under the MIT License. See the LICENSE file for more details.
