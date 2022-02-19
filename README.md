<h1 align="center">
             Medical Predictions Web App 🩺 💊 💉
</h1>
  
  ![image](https://user-images.githubusercontent.com/78029145/153434524-ca6c416b-3f8e-43ca-8174-6f68789209a5.png)


This app is used to predict the medical state of an individual
The disease sections include ->

**1. Covid-19**

**2. Diabetes**

**3. Heart Disease**

## Tech Stacks Used

<img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/>

## Libraries Used

<img src="https://img.shields.io/badge/numpy%20-%2314354C.svg?&style=for-the-badge&logo=numpy&logoColor=white"/> <img src="https://img.shields.io/badge/pandas%20-%2314354C.svg?&style=for-the-badge&logo=pandas&logoColor=white"/> <img src="https://img.shields.io/badge/plotly%20-%2314354C.svg?&style=for-the-badge&logo=plotly&logoColor=white"/>
<img src="https://img.shields.io/badge/streamlit%20-%2314354C.svg?&style=for-the-badge&logo=streamlit&logoColor=white"/> <img src="https://img.shields.io/badge/scikitlearn%20-%2314354C.svg?&style=for-the-badge&logo=scikitlearn&logoColor=white"/>

## Structure Of The Project

- Each prediction page is conneceted with a Machine Learning Model which uses Random Forest Classifier to predict the results.
- Also we have 3 different datasets used for each prediction.
- We can land into each prediction site of the web app from the options in the Navigation Menu.


- Each prediction is done with the help of 4 features which will be taken as input from the user.
- The most relevant features are taken into consideration for prediction also these features can be found out with simple tests or analysis without visiting any doctor.
- So the victim can get a broad overview of their health condition.

## The features taken into consideration

| Disease | Features |
| - | - |
| Covid-19 | Dry Cough, Fever, Sore Throat, Breathing Problem |
| Diabetes | Glucose, Insulin, Body Mass Index(BMI), Age |
| Heart Disease | Chest Pain, Blood Pressure(BP), Cholestrol, Max Heart Rate(HR) |

The feature selection is carefully done under the supervision of a medical science student.

## Deployment Of The Project

After the modeling part the model is deployed using Streamlit library on Streamlit Share so that the app is available for usage for everyone.

## Link To My Web Application -

https://share.streamlit.io/bhaswatiroy/medical-predictions-streamlit-web-app/main/app.py

## Link To My Video Explanation -

https://youtu.be/zv6ZIe4RbnQ

## Glance At The Hosted Application-

### 1. Home Page
![image](https://user-images.githubusercontent.com/78029145/153414617-663dfe56-18eb-47f5-8de6-54aa5d0e0351.png)

### 2. Covid-19 Page
![image](https://user-images.githubusercontent.com/78029145/153414774-e1c39199-db5e-4d40-9ed5-bd65edee1bf2.png)

### 3. Diabetes Page
![image](https://user-images.githubusercontent.com/78029145/153414946-8417cd23-6996-41c1-b989-8158cc980e43.png)

### 4. Heart Disease Page
![image](https://user-images.githubusercontent.com/78029145/153415061-38a25278-c48b-49f2-a6b8-be66817ec8c4.png)

### 5. Plots Page
![image](https://user-images.githubusercontent.com/78029145/153417820-dbfe50f7-1ce3-43ce-ac7d-4285d4d6f303.png)


