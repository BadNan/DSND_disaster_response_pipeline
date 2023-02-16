# DSND disaster response pipeline

## Summary:
A machine learning pipeline is built that classifies disaster messages. The pipeline loads and cleans data, tokenizes the messages, trains a model and evaluates its performances. The project is integrated into a web app where an individual can assess whether the message fits into different categories.

## Files in Repo:
The project contains the following files:

data folder: 

disaster_categories.csv: categories data
disaster_messages.csv: messages data
process_data.py: python file used to load and clean data

models folder:

train_classifier.py: python file used to create the ML pipeline

ETL Pipeline Preparation.ipynb: Notebook experiment for the ETL pipelines
ML Pipeline Preparation.ipynb: Notebook experiment for the machine learning pipelines
data/process_data.py: The ETL pipeline used to process data in preparation for model building.
models/train_classifier.py: The Machine Learning pipeline used to fit, tune, evaluate, and export the model to a Python pickle (pickle is not uploaded to the repo due to size constraints on github).
app/templates/~.html: HTML pages for the web app.
run.py: Start the Python server for the web app and prepare visualizations.
The app is now deployed on heroku at this link

Example message to classify: "Help, Fire!"

## How to run the Python scripts and web app
To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Run the following command in the app's directory to run your web app. python app.py
