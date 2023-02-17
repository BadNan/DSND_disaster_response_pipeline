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

app folder:

run.py: python fileto connect and start web app

## How to run the Python scripts and web app

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`
