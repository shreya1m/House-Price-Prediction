# House-Price-Prediction

## Overview
This project aims to predict house prices using machine learning techniques. The goal is to develop a model that can accurately estimate the price of a house based on various features such as crime rate, number of bedrooms, population, and other relevant factors. The project utilizes a dataset containing historical housing data, and the machine learning model is trained on this data to make predictions for new, unseen instances.

## Dataset
The dataset used for this project is sourced from house.csv. It includes various features such as:

    1. CRIM      per capita crime rate by town 
    2. ZN        proportion of residential land zoned for lots over 
                 25,000 sq. Ft.
    3. INDUS     proportion of non-retail business acres per town
    4. CHAS      Charles River dummy variable (= 1 if tract bounds 
                 river; 0 otherwise)
    5. NOX       nitric oxides concentration (parts per 10 million)
    6. RM        average number of rooms per dwelling
    7. AGE       proportion of owner-occupied units built prior to 1940
    8. DIS       weighted distances to five Boston employment centres
    9. RAD       index of accessibility to radial highways
    10. TAX      full-value property-tax rate per $10,000
    11. PTRATIO  pupil-teacher ratio by town
    12. B        1000(Bk - 0.63) ^2 where Bk is the proportion of blacks 
                 by town
    13. LSTAT    % lower status of the population.
    14. MEDV     Median value of owner-occupied homes in $1000's

## Project Structure
The project is organized into the following directories:

- data: Contains the dataset used for training and testing the machine learning model.

- src: Python scripts containing modular code for data preprocessing, feature engineering, and model training. This promotes code reusability and maintainability.

- models: Saved machine learning models in Joblib. These models can be loaded and used for making predictions on new data.

## Dependencies
- Python 3.x
- Libraries: NumPy, Pandas, Scikit-Learn, Matplotlib, Seaborn, etc. (provide a comprehensive list in a requirements.txt file)
## Usage
### Clone the repository:

git clone https://github.com/shreya1m/House-Price-Prediction.git
### Navigate to the project directory:
cd house-price-prediction
### Install dependencies:
- pip install -r requirements.txt
- Execute python script.

## Future Enhancements
- Fine-tuning hyperparameters to improve model performance.
- Exploring additional features for better prediction accuracy.
- Deploying the model as a web application or API for real-time predictions.


Feel free to contribute, open issues, or provide feedback to make this project more robust and effective in predicting house prices.





