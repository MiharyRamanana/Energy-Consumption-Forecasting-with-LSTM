# Energy Consumption Forecasting with LSTM

## Project Intro/Objective
The purpose of this project is to estimate the average daily consumption of energy of french customers using time series. It could help companies to predict future energy needs and ensure a reliable power supply.


### Methods Used
* Deep Learning Model : LSTM
* Data Visualization
* Predictive Modeling
* Feature Engineering
* Early Stopping and Learning Rate Scheduling

### Technologies
* Python
* Pandas, jupyter
* Tensorflow, Keras
* Numpy, Matplotlib, Seaborn

## Project Description
For this project we use the dataset from Enedis Open Data [bilan electrique jour](https://data.enedis.fr/explore/dataset/bilan-electrique-jour). This dataset contains the average daily power consumption and production by different categories of client. We focus here on the private individuals' consumption from 2020-03-08 to 2025-03-07. As we want to work with an LSTM model, we implement lag features and rolling windows from this dataset. This will help the model to understand patterns and seasonality.
We know that energy consumption for private individuals depends on the weather. So we add another dataset [weather in France](https://www.data.gouv.fr/fr/datasets/observation-meteorologique-historiques-france-synop/) and use different features from this dataset to train our model and get better results. This dataset focuses mainly on the weather in Toulouse, Haute-Garonne and we use the temperature, humidity, speed of wind and rain as features for our model from 2020-03-08 to 2025-03-07.
At first, the weather dataset contained 14518 data due to the fact that data were collected every 3 hours everyday, so we implement a daily weather dataset and take the daily mean of the different features.

For the LSTM model, we use 60-length sequences to train the model. 
Here is the structure of our model :
![image](https://github.com/user-attachments/assets/fe5266eb-bcb5-4d8a-b338-a131d8d3318c)

To have an efficient training, we use early stopping and learning rate scheduling.

![image](https://github.com/user-attachments/assets/2788005e-a5e7-490f-9237-2af6a79dbd77)

When trying to predict the energy consumption, we obtain the following results :
Mean Squared Error: 511472321823467.7
Mean Absolute Error: 15545128.062976195
R-squared Score: 0.9663764453684025

Overall, the results obtained are good, especially the R². For a global prediction requirement, the model is acceptable.

To increase accuracy, i.e. reduce MAE, adjustments are possible, such as hyperparameter optimization (Dropout, BatchNorm).
