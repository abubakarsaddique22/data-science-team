# Car Sales Data Science Project

An end-to-end Data Science project for analyzing and predicting car prices using a real-world dataset. This project demonstrates the complete data science workflow, from data cleaning and exploration to model building, evaluation, and deployment-ready insights.

## Dataset

The dataset contains information about used cars, with the following features:

| Feature        | Description |
|----------------|-------------|
| `name`         | Car model name |
| `year`         | Year of manufacture |
| `selling_price`| Selling price of the car (target variable) |
| `km_driven`    | Distance driven in kilometers |
| `fuel`         | Fuel type (Petrol, Diesel, CNG, etc.) |
| `seller_type`  | Seller type (Individual, Dealer, etc.) |
| `transmission` | Transmission type (Manual / Automatic) |
| `owner`        | Number of previous owners |

## Project Workflow

This project follows a typical data science pipeline:

1. **Data Cleaning**
   - Handling missing values
   - Removing duplicates
   - Correcting data types
   - Standardizing categorical features (e.g., `fuel`, `transmission`)

2. **Exploratory Data Analysis (EDA)**
   - Summary statistics
   - Distribution plots for numerical features
   - Count plots for categorical features
   - Correlation analysis to understand relationships between variables

3. **Data Visualization**
   - Histograms, boxplots, and scatter plots
   - Heatmap for correlation
   - Insights on top-selling cars, fuel types, and price trends

4. **Feature Engineering**
   - Creating derived features if necessary (e.g., car age = current year - `year`)
   - Encoding categorical variables (One-Hot Encoding / Label Encoding)
   - Scaling numerical features if required

5. **Model Building**
   - Splitting data into training and testing sets
   - Testing multiple regression models (Linear Regression, Random Forest, XGBoost)
   - Hyperparameter tuning for optimal performance

6. **Model Evaluation**
   - Metrics: RMSE, MAE, R² Score
   - Comparing different models to select the best one
   - Residual analysis and error distribution

7. **Prediction & Deployment**
   - Users can provide new car details as input
   - Model predicts the expected selling price
   - Ready for integration into a web or desktop application

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/abubakarsaddique22/data-science-team.git
cd data-science-team
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Place your CSV dataset in the `data/` folder and update the file path in `main.py` or your script.

4. Run the project:

```bash
python main.py
```

5. Follow prompts to:
   - Clean the dataset
   - Explore & visualize data
   - Build and evaluate predictive models
   - Get predictions on new input data

## Technologies Used

- Python 3.x
- Pandas, NumPy (Data manipulation)
- Matplotlib, Seaborn (Visualization)
- Scikit-learn, XGBoost (Model building)
- Jupyter Notebook / Python scripts

## Project Highlights

- Fully automated data science pipeline from raw CSV to predictions
- Interactive prompts to guide user through cleaning, analysis, and modeling
- Includes EDA, visualization, feature engineering, model building, and evaluation

## License

This project is licensed under MIT License.

## Explore

Check out the full project on [GitHub](https://github.com/abubakarsaddique22/data-science-team)

