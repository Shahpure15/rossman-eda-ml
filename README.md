# Rossmann Store Sales - Complete Data Science Pipeline

This project implements a complete data science pipeline for the Rossmann Store Sales dataset, including data preprocessing, exploratory data analysis, machine learning modeling, and an interactive dashboard.

## 📊 Project Overview

The Rossmann Store Sales dataset contains historical sales data for 1,115 Rossmann stores. The goal is to predict daily sales and provide insights into factors affecting store performance.

## 🗂️ Project Structure

```
rossmann-eda-ml/
│
├── data/
│   ├── raw/                     # Original dataset files
│   │   ├── train.csv
│   │   ├── store.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   └── processed/               # Cleaned and processed data
│
├── notebooks/
│   ├── data_preprocessing.ipynb # Data cleaning and feature engineering
│   └── eda.ipynb                # Exploratory data analysis
│
├── models/
│   └── sales_prediction.py      # Machine learning model
│
├── dashboard/
│   └── app.py                   # Interactive Streamlit dashboard
│
├── outputs/
│   └── plots/                   # Generated visualizations
│
├── docs/
│   └── synopsis.md              # Project summary and insights
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run data preprocessing:**
   ```bash
   jupyter notebook notebooks/data_preprocessing.ipynb
   ```

3. **Explore the data:**
   ```bash
   jupyter notebook notebooks/eda.ipynb
   ```

4. **Train the model:**
   ```bash
   python models/sales_prediction.py
   ```

5. **Launch the dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```

## 📈 Key Features

- **Data Preprocessing**: Complete data cleaning, feature engineering, and categorical encoding
- **Exploratory Data Analysis**: Comprehensive visualizations and statistical analysis
- **Machine Learning**: Sales prediction using Random Forest and Linear Regression
- **Interactive Dashboard**: Streamlit app for exploring sales patterns
- **Version Control**: Proper Git workflow with meaningful commits

## 📊 Dataset Information

**Train Dataset:**
- 1,017,209 records
- 9 features including Store, Sales, Customers, Date, Promo, etc.

**Store Dataset:**
- 1,115 stores
- Store metadata including type, assortment, competition info

## 🔍 Key Insights

See `docs/synopsis.md` for detailed analysis and business insights.

## 📝 Usage Examples

### Loading Processed Data
```python
import pandas as pd
df = pd.read_csv('data/processed/train_processed.csv')
```

### Running Predictions
```python
from models.sales_prediction import SalesPredictor
predictor = SalesPredictor()
predictions = predictor.predict(test_data)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🏆 Acknowledgments

- Rossmann for providing the dataset
- Kaggle for hosting the competition
- The open-source community for the amazing tools used in this project
