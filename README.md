# Bioreactor-Protein-Yield-Prediction-System
**Bioreactor Protein Yield Prediction System**

A complete machine learning pipeline for predicting protein yield in bioreactor processes, including synthetic data generation, automated data cleansing, and Random Forest regression modeling.

**📋 Project Overview**

This project consists of two main components:

1\. **\*\*\`Dataset\_creation.py\`\*\*** - Generates synthetic bioreactor batch data with controlled parameters

2\. **\*\*\`Model\_generation.py\`\*\*** - Builds a prediction model with automated data cleansing and evaluation

The system demonstrates a complete ML workflow from data generation to model deployment, specifically designed for bioprocess engineering applications.

**🚀 Installation**

**Prerequisites**

\- Python 3.7 or higher

\- pip package manager

**Step 1: Install Python**

This project requires Python 3.7 or higher. Check if Python is installed:

\`\`\`bash

python --version

**If Python is not installed:**

**In you Windows PC:**

1.  Download Python from [python.org](https://www.python.org/downloads/)
2.  Run the installer
3.  **IMPORTANT**: Check "Add Python to PATH" during installation
4.  Verify installation:

bash

python --version

**Step 2: Set Up Virtual Environment**

Create and activate a virtual environment to isolate project dependencies:

**In Windows (PowerShell):**

bash

_\# Create virtual environment_

python -m venv venv

_\# Activate virtual environment_

.\\venv\\Scripts\\Activate.ps1

**Step 3: Install Dependencies**

With the virtual environment activated, install required packages:

bash

_\# Upgrade pip first_

python -m pip install --upgrade pip

_\# Install all required packages_

pip install numpy pandas matplotlib scikit-learn joblib

**Alternative: Install using requirements.txt:**

bash

pip install -r requirements.txt

**requirements.txt content:**

txt

numpy>=1.19.0

pandas>=1.2.0

matplotlib>=3.3.0

scikit-learn>=0.24.0

joblib>=1.0.0

**Step 4: Verify Installation**

Create a test script to verify all packages are installed correctly:

python

_\# test\_installation.py_

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn

import joblib

print("✅ NumPy version:", np.\_\_version\_\_)

print("✅ Pandas version:", pd.\_\_version\_\_)

print("✅ Matplotlib version:", plt.matplotlib.\_\_version\_\_)

print("✅ Scikit-learn version:", sklearn.\_\_version\_\_)

print("✅ Joblib version:", joblib.\_\_version\_\_)

print("\\n🎉 All packages installed successfully!")

Run the test:

bash

python test\_installation.py

**Step 5: Download Project Files**

Create a new folder for your project and add the Python scripts - Dataset\_creation.py and Model\_generation.py. You can also create a new directory in windows powershell with this command - mkdir <directory name>

**💻 Usage**

**Step 1: Generate Synthetic Dataset**

In windows power shell, navigate to the created folder, containing the two python scripts, via cd command:

cd path to the folder

Run the first python script:

python Dataset\_creation.py

This creates bioreactor\_batch\_data.csv with 5 batches, 50 timepoints each, and different feed rates.

**Expected output:**

*   250 rows of bioreactor process data
*   Visualization saved as simple\_bioreactor\_data.png
*   CSV file with columns: Batch\_ID, Feed\_Rate\_gLh, Time\_h, Temperature\_C, pH, DO\_percent, Protein\_Yield\_gL

**Step 2: Train the Prediction Model**

bash

python Model\_generation.py

**What happens during model training:**

1.  Loads the generated dataset
2.  Intentionally adds data quality issues (missing values, outliers)
3.  Performs automated data cleansing
4.  Trains a Random Forest Regressor
5.  Evaluates model performance
6.  Generates visualization plots
7.  Saves model and predictions

**Output files generated:**

| File | Description |
| --- | --- |
| bioreactor_data_cleaned.csv | Cleansed dataset after outlier treatment |
| bioreactor_yield_model.pkl | Trained model for future predictions |
| test_predictions.csv | Test set predictions with residuals |
| model_performance_with_cleansing.png | 4-panel performance visualization |

**🧹 Data Cleansing Features**

The model automatically handles common data quality issues:

**1\. Missing Value Treatment**

*   Detects NaN values in all columns
*   Imputes missing values with column median
*   Handles completely empty columns gracefully

**2\. Outlier Detection & Capping**

Applies bioprocess-specific constraints:

| Parameter | Valid Range | Treatment |
| --- | --- | --- |
| Temperature | 35 - 39°C | Cap outliers |
| pH | 6.5 - 7.5 | Cap outliers |
| Dissolved Oxygen | 30 - 100% | Cap outliers |
| Feed Rate | 0.3 - 2.5 g/L/hr | Cap outliers |
| Protein Yield | 0 - 15 g/L | Cap outliers |

**3\. Duplicate Removal**

*   Automatically identifies and removes duplicate rows
*   Resets index after cleansing

**🎯 Model Performance**

The Random Forest model achieves:

| Metric | Typical Value | Interpretation |
| --- | --- | --- |
| R² Score | 0.90 - 0.95 | Excellent - explains >90% variance |
| RMSE | 0.50 - 0.70 g/L | Good - ~15-20% relative error |
| MAE | 0.40 - 0.60 g/L | Good - average prediction error |

**Feature Importance (Typical)**

1.  **Time\_h** (45-55%) - Most influential
2.  **Feed\_Rate\_gLh** (25-35%) - Key process parameter
3.  **Temperature\_C** (5-10%) - Moderate influence
4.  **pH** (3-7%) - Minor influence
5.  **DO\_percent** (2-5%) - Least influence

**🔧 Model Configuration**

python

RandomForestRegressor(

n\_estimators=100, _\# Number of decision trees_

random\_state=42, _\# Reproducibility_

n\_jobs=-1 _\# Use all CPU cores_

)

**🆕 Making Predictions on New Data**

python

import joblib

import pandas as pd

_\# Load the trained model_

model = joblib.load('bioreactor\_yield\_model.pkl')

_\# Prepare new batch conditions_

new\_batch = pd.DataFrame({

'Feed\_Rate\_gLh': \[1.4\],

'Time\_h': \[30\],

'Temperature\_C': \[37.1\],

'pH': \[6.9\],

'DO\_percent': \[72\]

})

_\# Predict yield_

prediction = model.predict(new\_batch)\[0\]

print(f"Predicted Protein Yield: {prediction:.3f} g/L")

**📈 Visualization Outputs**

**1\. Dataset Visualization (simple\_bioreactor\_data.png)**

*   Protein yield over time for each batch
*   Feed rate vs final yield correlation
*   Temperature distribution by batch
*   pH distribution by batch

**2\. Model Performance (model\_performance\_with\_cleansing.png)**

*   Actual vs predicted scatter plot with R² score
*   Residual plot with RMSE
*   Feature importance horizontal bar chart
*   Prediction distribution histogram

**⚠️ Challenges Faced and Solutions**

**Challenge 1: Missing Value Handling Warnings**

**Problem**: Pandas FutureWarning about inplace=True operation on DataFrame slice.

**Solution**:

python

_\# Before (caused warning)_

df\_clean\[col\].fillna(median\_val, inplace=True)

_\# After (no warning)_

df\_clean\[col\] = df\_clean\[col\].fillna(median\_val)

**Challenge 2: Completely Empty Columns**

**Problem**: Artificially added columns ('Temperature', 'DO', 'Yield', 'Feed\_Rate') contained all NaN values, causing median calculation to fail.

**Solution**:

*   Added check for columns with all NaN values
*   Fill completely empty columns with 0 instead of median
*   Drop artificial columns after cleansing

**Challenge 3: Outlier Capping Impact**

**Problem**: Aggressive outlier capping removed valid biological variation.

**Solution**:

*   Defined bioprocess-specific realistic ranges based on literature
*   Implemented capping instead of removal to preserve data points
*   Added outlier count tracking for transparency

**Challenge 4: Model Overfitting Concerns**

**Problem**: High R² but moderate RMSE raised overfitting questions.

**Solution**:

*   Added training vs test metrics comparison
*   Implemented cross-validation readiness
*   Documented expected performance benchmarks for bioprocess data
*   Added relative error metrics (% of mean and range)

**📁 Project Structure**

text

bioreactor-yield-prediction/

│

├── Dataset\_creation.py # Synthetic data generator

├── Model\_generation.py # ML model with cleansing

├── requirements.txt # Package dependencies

├── README.md # Project documentation

│

├── bioreactor\_batch\_data.csv # Generated dataset

├── bioreactor\_data\_cleaned.csv # Cleansed dataset

├── bioreactor\_yield\_model.pkl # Trained model

├── test\_predictions.csv # Test set predictions

│

├── simple\_bioreactor\_data.png # Dataset visualization

└── model\_performance\_with\_cleansing.png # Model performance plots

**🔬 Future Improvements**

1.  **Cross-Validation**: Implement k-fold cross-validation for more robust evaluation
2.  **Additional Features**: Include agitation speed, pressure, CO2 levels
3.  **Deep Learning**: Experiment with LSTM networks for time-series prediction
4.  **API Deployment**: Create Flask/FastAPI endpoint for real-time predictions
5.  **Dashboard**: Build interactive dashboard for visualization
6.  **Automated Reporting**: Generate PDF reports with model performance metrics

**📄 License**

This project is licensed under the MIT License - see the LICENSE file for details.

**🙏 Acknowledgments**

*   Scikit-learn documentation and community
*   Bioprocess engineering principles and domain experts
*   Open source contributors and maintainers