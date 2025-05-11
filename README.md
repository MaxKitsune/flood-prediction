# Flood Prediction: A machine learning university project

## Project Overview

This repository contains the work for a university project in the "Machine Learning" course. The project was a **collaborative effort**, focusing on applying **various machine learning models** to a dataset for predictive analysis, specifically targeting flood prediction.

The primary goals included **data preparation**, which involved various **mathematical and linear methods** for cleaning, transforming, and feature engineering. Subsequently, different models such as **linear regression, regression trees, and neural networks** were implemented and **evaluated** for their performance in predicting flood-related outcomes.

**Author's Note:** My specific contributions to this project involved the data preparation phase, the development and analysis of the regression tree model, and the implementation and exploration of the neural network models.
Additionally, the comments and other documentation in the code were translated from German to English. Some translation errors may still be present.

## Project Structure

The repository is organized as follows:

*   `1_data_preparation.ipynb`: Jupyter Notebook detailing the initial data loading, cleaning, application of mathematical/linear preprocessing techniques (e.g., normalization, feature scaling), and exploratory data analysis (EDA).
*   `2_Linear_Model.ipynb`: Jupyter Notebook covering the implementation, training, and evaluation of a linear regression model.
*   `3_regression_tree.ipynb`: Jupyter Notebook focused on the regression tree model, including its development, tuning, and performance analysis.
*   `4_train_nn.py`: Python script used for training the neural network model.
*   `5_Neuronales_Netz.ipynb`: Jupyter Notebook for the analysis, evaluation, and fine-tuning of the trained neural network. (Note: "Neuronales Netz" is German for Neural Network).
*   `model_results.csv`: A CSV file storing the performance metrics and results from the different models evaluated.

## Tech Stack

The project primarily utilizes the following technologies:

*   **Python:** The core programming language used.
*   **Jupyter Notebooks:** For interactive development, analysis, and documentation.
*   **Pandas:** For data manipulation and analysis.
*   **NumPy:** For numerical computations.
*   **Scikit-learn:** For implementing linear models, regression trees, and other machine learning utilities (e.g., metrics, model selection).
*   **TensorFlow/Keras or PyTorch:** (Assumed for the neural network implementation) For building and training neural networks.
*   **External AI GPU Server:** Used to run intensive training tasks and reduce experiment turnaround times.

## Key Learnings

This project provided valuable experience in:

*   **End-to-End ML Workflow:** From data collection and preprocessing to model selection, training, evaluation, and interpretation of results.
*   **Data Preprocessing:** Techniques for cleaning, transforming, and preparing data for machine learning models.
*   **Model Implementation:** Hands-on experience with implementing and configuring different types of machine learning algorithms:
    *   Linear Regression
    *   Decision/Regression Trees
    *   Neural Networks
*   **Hyperparameter Tuning:** Understanding the impact of different hyperparameters and methods to optimize model performance.
*   **Model Evaluation & Comparison:** Using appropriate metrics to assess model performance and compare the effectiveness of different approaches.
*   **Collaborative Development:** Working in a team on a shared codebase and project goals.
*   **Scientific Computing Libraries:** Proficiency in using Python libraries essential for data science and machine learning.


## Data

The raw data files used for training the models (`Flut-Daten.csv`, `pr_hourly_DWD_ID1550.dat`, and `Q_hourly_ID16425004`) are not included in this repository due to their size or other constraints. However, they can be provided upon request.

---

*This README was generated with assistance from an AI coding assistant.* 
