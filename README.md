# Predictive Model for Studying Air Quality and Allergic Subject Symptoms

## Overview
This thesis focuses on the development of a predictive model to explore the relationships between air quality and symptoms in allergic individuals. Utilizing machine learning techniques, particularly a Long Short-Term Memory (LSTM) neural network model, we aimed to understand how air pollutant concentrations impact allergic reactions and symptoms.

## Dataset
The dataset comprises hourly measurements of air pollutants from a specific urban area, coupled with symptom frequency data collected from allergic individuals through surveys. Preprocessing techniques like oversampling (SMOTE) and clustering (K-Means) were employed to address data imbalances and identify meaningful patterns.

## Key Findings
- Identification of significant correlations between specific air pollutants (NOx, PM10) and the worsening of allergic symptoms.
- Successful application of LSTM to predict symptom exacerbation based on air quality metrics.
- Insights into effective data preprocessing techniques to enhance model performance in the presence of unbalanced datasets.

## Technologies Used
- Python for data analysis and machine learning model development.
- Pandas and NumPy for data manipulation.
- Matplotlib and Seaborn for data visualization.
- Scikit-learn for implementing machine learning algorithms and preprocessing techniques.
- TensorFlow/Keras for building and training the LSTM neural network model.

## Model Training and Evaluation
The LSTM model was trained using a split of the preprocessed dataset, with evaluation based on its ability to accurately predict allergic symptom exacerbation from air quality data. Metrics such as RMSE (Root Mean Square Error) and MAE (Mean Absolute Error) were used to assess model performance.


## Acknowledgements
- Survey participants for contributing valuable data on allergic symptoms.
