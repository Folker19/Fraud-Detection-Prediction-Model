# Fraud Detection Prediction Model
End to end EDA, CDA and ML analysis over a synthetic e-commerce transaction's dataset with the objective of training the most accurate model for fraud detection purposes.

## Repository Structure

- `data/`: Directory to store the dataset files. 
- `src/models`: Source code for the project.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and prototyping.
- `graphs/`: Directory to store generated plots and graphs.
- `models/`: Directory to save trained models and transformers.
- `requirements.txt`: List of required Python packages.
- `README.md`: This file.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Required Python packages listed in `requirements.txt`

### Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/fraud-detection-model.git
    ```

2. Navigate to the project directory:

    ```sh
    cd fraud-detection-model
    ```

3. Create and activate a virtual environment:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

### Usage

1. **Data Preprocessing**: Prepare the dataset by encoding categorical variables.

    ```sh
    python src/preprocessing.py
    ```

    This script will load the dataset, encode the categorical variables, and save the processed data.

2. **Sampling**: Balance the dataset by undersampling legitimate transactions.

    ```sh
    python src/sampling.py
    ```

    This script will balance the dataset and save the balanced data.

3. **Model Training**: Train the machine learning model using the balanced dataset.

    ```sh
    python src/train_model.py
    ```

    This script will train the model and save the trained model and transformer.

4. **Model Evaluation**: Evaluate the trained model on the test set.

    ```sh
    python src/evaluate_model.py
    ```

    This script will load the trained model, make predictions on the test set, and print the evaluation metrics.

## Files Explanation

- **data**: Directory where you should place your dataset files.
- **src/preprocessing.py**: Script for preprocessing the data. This includes encoding categorical variables and saving the processed data.
- **src/sampling.py**: Script for balancing the dataset by undersampling legitimate transactions.
- **src/train_model.py**: Script for training the machine learning model.
- **src/evaluate_model.py**: Script for evaluating the trained model on the test set.
- **src/utils.py**: Contains utility functions used in various scripts.
- **notebooks**: Jupyter notebooks for exploratory data analysis and prototyping.
- **graphs**: Directory to store generated plots and graphs.
- **models**: Directory to save trained models and transformers.
- **requirements.txt**: List of required Python packages.
- **README.md**: Provides an overview of the repository, installation instructions, and usage guidelines.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
