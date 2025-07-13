import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

class SalesPredictor:
    """
    A comprehensive sales prediction model for Rossmann stores.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.metrics = {}
        
    def load_data(self, filepath):
        """Load and prepare the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(filepath)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        print(f"Dataset loaded: {self.df.shape}")
        return self.df
    
    def prepare_features(self, df=None):
        """Prepare features for modeling"""
        if df is None:
            df = self.df.copy()
            
        print("Preparing features...")
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Extract date features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['IsWeekend'] = (df['DayOfWeek'] >= 6).astype(int)
        
        # Load and merge store data
        try:
            store_data = pd.read_csv('../data/raw/store.csv')
            df = pd.merge(df, store_data, on='Store', how='left')
        except FileNotFoundError:
            print("Store data not found, using basic features only")
        
        # Handle missing values
        df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].median())
        df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0)
        df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0)
        df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)
        df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0)
        df['PromoInterval'] = df['PromoInterval'].fillna('None')
        
        # Create competition duration feature
        df['CompetitionDuration'] = np.where(
            (df['CompetitionOpenSinceYear'] > 0) & (df['CompetitionOpenSinceMonth'] > 0),
            (df['Year'] - df['CompetitionOpenSinceYear']) * 12 + (df['Month'] - df['CompetitionOpenSinceMonth']),
            0
        )
        
        # Create promo2 duration feature
        df['Promo2Duration'] = np.where(
            (df['Promo2SinceYear'] > 0) & (df['Promo2SinceWeek'] > 0),
            (df['Year'] - df['Promo2SinceYear']) * 52 + (df['WeekOfYear'] - df['Promo2SinceWeek']),
            0
        )
        
        # Create sales per customer feature
        df['SalesPerCustomer'] = df['Sales'] / df['Customers'].replace(0, 1)
        
        # Store performance features
        store_avg_sales = df.groupby('Store')['Sales'].mean()
        df['StoreAvgSales'] = df['Store'].map(store_avg_sales)
        
        # Define feature columns
        self.numeric_features = [
            'Store', 'DayOfWeek', 'Open', 'Promo', 'SchoolHoliday',
            'Month', 'Year', 'Day', 'WeekOfYear', 'IsWeekend',
            'Customers', 'CompetitionDistance', 'CompetitionDuration',
            'Promo2Duration', 'SalesPerCustomer', 'StoreAvgSales'
        ]
        
        self.categorical_features = [
            'StoreType', 'Assortment', 'StateHoliday'
        ]
        
        # Keep only features that exist in the dataframe
        self.numeric_features = [f for f in self.numeric_features if f in df.columns]
        self.categorical_features = [f for f in self.categorical_features if f in df.columns]
        
        # Handle missing values in numeric features
        for col in self.numeric_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Encode categorical variables
        df_encoded = df.copy()
        for col in self.categorical_features:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = le
        
        # Combine all features
        all_features = self.numeric_features + self.categorical_features
        X = df_encoded[all_features]
        y = df_encoded['Sales']
        
        print(f"Features prepared: {X.shape[1]} features")
        print(f"Sample size: {X.shape[0]} records")
        return X, y, all_features
    
    def train_models(self, X, y):
        """Train multiple models and compare performance"""
        print("Training models...")
        
        # Sample the data for faster training (use 10% of data)
        sample_size = min(100000, len(X))
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[sample_indices]
        y_sample = y.iloc[sample_indices]
        
        print(f"Using sample size: {sample_size} records for training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_sample, y_sample, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        # Define models with reduced complexity
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=50, 
                max_depth=15, 
                random_state=42, 
                n_jobs=1,
                max_features='sqrt'
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=50, 
                max_depth=6, 
                random_state=42
            )
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            print(f"  Training {name}...")
            
            # Use scaled features for Linear Regression, original for tree-based models
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'model': model
            }
            
            # Store model
            self.models[name] = model
            
            print(f"    RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}")
        
        # Select best model based on R2 score
        best_model_name = max(results, key=lambda x: results[x]['R2'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        self.metrics = results
        
        print(f"Best model: {best_model_name}")
        
        # Feature importance for tree-based models
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = dict(zip(X.columns, self.best_model.feature_importances_))
            
        return results, X_test, y_test
    
    def plot_results(self, results, X_test, y_test):
        """Plot model results and feature importance"""
        print("Generating visualizations...")
        
        # Model comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Model comparison metrics
        metrics_df = pd.DataFrame(results).T
        metrics_df[['RMSE', 'MAE']].plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Model Comparison - RMSE & MAE')
        axes[0, 0].set_ylabel('Error')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. R2 scores
        r2_scores = [results[model]['R2'] for model in results]
        axes[0, 1].bar(results.keys(), r2_scores, color=['blue', 'green', 'red'])
        axes[0, 1].set_title('Model Comparison - R² Score')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Actual vs Predicted (best model)
        if self.best_model_name == 'Linear Regression':
            y_pred = self.best_model.predict(self.scalers['standard'].transform(X_test))
        else:
            y_pred = self.best_model.predict(X_test)
            
        axes[1, 0].scatter(y_test, y_pred, alpha=0.5)
        axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Sales')
        axes[1, 0].set_ylabel('Predicted Sales')
        axes[1, 0].set_title(f'Actual vs Predicted ({self.best_model_name})')
        
        # 4. Feature importance (if available)
        if self.feature_importance:
            top_features = dict(sorted(self.feature_importance.items(), 
                                     key=lambda x: x[1], reverse=True)[:10])
            axes[1, 1].barh(list(top_features.keys()), list(top_features.values()))
            axes[1, 1].set_title('Top 10 Feature Importance')
            axes[1, 1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('../outputs/plots/model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Residual analysis
        residuals = y_test - y_pred
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predicted Sales')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        
        plt.tight_layout()
        plt.savefig('../outputs/plots/residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath='../models/sales_prediction_model.joblib'):
        """Save the trained model"""
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_importance': self.feature_importance,
            'metrics': self.metrics
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to: {filepath}")
    
    def predict(self, X):
        """Make predictions on new data"""
        if self.best_model_name == 'Linear Regression':
            return self.best_model.predict(self.scalers['standard'].transform(X))
        else:
            return self.best_model.predict(X)
    
    def generate_report(self):
        """Generate a comprehensive model report"""
        print("\n" + "="*60)
        print("ROSSMANN SALES PREDICTION MODEL REPORT")
        print("="*60)
        
        print(f"\nBest Model: {self.best_model_name}")
        print(f"Model Performance:")
        
        best_metrics = self.metrics[self.best_model_name]
        print(f"  • RMSE: €{best_metrics['RMSE']:,.2f}")
        print(f"  • MAE: €{best_metrics['MAE']:,.2f}")
        print(f"  • R² Score: {best_metrics['R2']:.3f}")
        print(f"  • Model explains {best_metrics['R2']*100:.1f}% of sales variance")
        
        print(f"\nAll Models Performance:")
        for model, metrics in self.metrics.items():
            print(f"  {model}:")
            print(f"    RMSE: €{metrics['RMSE']:,.2f} | MAE: €{metrics['MAE']:,.2f} | R²: {metrics['R2']:.3f}")
        
        if self.feature_importance:
            print(f"\nTop 10 Most Important Features:")
            sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            for i, (feature, importance) in enumerate(sorted_features[:10], 1):
                print(f"  {i:2d}. {feature}: {importance:.4f}")
        
        print(f"\nBusiness Insights:")
        print(f"  • Model can predict sales with ±€{best_metrics['MAE']:,.0f} average error")
        print(f"  • Suitable for: Daily sales forecasting, inventory planning, budget allocation")
        print(f"  • Recommendation: Use for stores with similar characteristics to training data")
        
        print(f"\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

def main():
    """Main execution function"""
    print("ROSSMANN SALES PREDICTION MODEL TRAINING")
    print("="*50)
    
    # Initialize predictor
    predictor = SalesPredictor()
    
    # Load data
    try:
        df = predictor.load_data('../data/raw/train.csv')
    except FileNotFoundError:
        print("Dataset not found. Please ensure the data file exists.")
        return
    
    # Prepare features
    X, y, features = predictor.prepare_features(df)
    
    # Train models
    results, X_test, y_test = predictor.train_models(X, y)
    
    # Plot results
    predictor.plot_results(results, X_test, y_test)
    
    # Save model
    predictor.save_model()
    
    # Generate report
    predictor.generate_report()
    
    print("\nModel training completed successfully!")
    print("Check 'outputs/plots/' for visualizations")
    print("Model saved in 'models/' directory")

if __name__ == "__main__":
    main()
