# Counter-Strike Round Winner Prediction

## Description

This project focuses on building and comparing multiple machine learning algorithms to predict the winners of rounds in Counter-Strike: Global Offensive (CS:GO).

## Objective

The objective is to classify the winners of each round based on in-game attributes such as weapons, economy, maps, teamwork, and competitive play.

## Dataset Description

The dataset consists of in-game attributes such as weapons, economy, maps, teamwork, and competitive play.

## Steps Involved

1. **Data Loading**: Load the dataset.
2. **Exploratory Data Analysis (EDA)**: Analyze the dataset to understand its structure and characteristics.
3. **Data Preprocessing**: Preprocess the data by handling missing values, duplicates, and encoding categorical variables.
4. **Model Building**: Build machine learning models using various algorithms such as Logistic Regression, Decision Tree Classifier, and Random Forest Classifier.
5. **Model Evaluation and Comparison**: Evaluate the models using metrics like Accuracy, Precision, Recall, F1-score, and ROC-AUC. Compare the performance of different models.
6. **Model Saving and Loading**: Save the best performing model for future use.

## Libraries Used

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Files Included

- [cs-go.csv](https://drive.google.com/file/d/1hv3ui6xtPM_hDyVwwm1OaOH-cq9bLyJ3/view?usp=drive_link): Dataset containing in-game attributes.
- [random_forest_model.pkl](https://drive.google.com/file/d/1HEy7gEwStU3p1X3c1fIs78g-7jFNgMjD/view?usp=drive_link): Saved Random Forest Classifier model.

## Usage

1. Clone the repository: `git clone https://github.com/adilbhartiya/Counter-Strike-Round-Winner-Prediction.git`
2. Navigate to the project directory: `cd Counter-Strike-Round-Winner-Prediction`
3. Open and run `CS-GO Round Winner.ipynb` in Jupyter Notebook or any compatible environment.

## Conclusion

The Random Forest Classifier emerged as the top-performing model with robust performance in predicting round winners in CS:GO. Further optimization and fine-tuning of the model could enhance its performance even more.

For detailed implementation and analysis, refer to the Jupyter Notebook (`CS-GO Round Winner.ipynb`) in this repository.
