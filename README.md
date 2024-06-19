# Fraud Detection Prediction Model

An end-to-end Exploratory Data Analysis (EDA), Confirmatory Data Analysis (CDA), and Machine Learning (ML) analysis of a synthetic e-commerce transaction dataset to train an accurate fraud detection model. This project serves academic purposes, hence the extended explanations within the notebooks.

## Repository Structure

- `.streamlit/`: Streamlit palette settings file.
- `data/`: Directory to store the dataset files.
- `graphs/`: Directory to store generated plots and graphs.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and prototyping.
- `src/models`: Directory to save trained models and transformers.
- `LICENSE`: The license under which this document is licensed.
- `README.md`: This file.
- `requirements.txt`: Required packages to install for running the code.
- `Streamlit_FraudDetection.py`: Script to run the Streamlit app.

## Getting Started

### Prerequisites

- Python 3.10 recommended.
- Required Python packages listed in `requirements.txt`.

### Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/fraud-detection-model.git
    ```

2. Navigate to the project directory:

    ```sh
    cd Fraud-Detection-Prediction-Model
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

5. Extract the compressed files:

    Given the size of the dataset, some files have been compressed in .rar format. Extract them before running the notebooks.


### Usage

1. **Data Preprocessing**: Run the notebook for data cleaning and preprocessing. A `transactions_processed.csv` file will be generated.

2. **EDA and CDA**: Run the notebooks for EDA and CDA analysis. Some graphs will be stored in the `graphs/` directory.

3. **Model Training**:
   - **ML_percentile_sampling.ipynb**: Run the notebook for data transformation, balancing, and sampling using a mathematical approach. The best models will be stored according to the different ML techniques applied.
   - **ML_random_sampling.ipynb**: Run the notebook for data transformation, balancing, and sampling using a random approach. The best models will be stored according to the different ML techniques applied.

4. **Model Evaluation**: Run the Streamlit app with the following command:

    ```sh
    streamlit run Streamlit_FraudDetection.py
    ```

    This will deploy the model on a local host.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Usage

Feel free to use this project as a reference for data analysis or ML models. If you leverage this project for your own research or projects, please provide proper attribution.

## License

This project is licensed under the MIT License.
