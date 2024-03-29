# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Files:
- **app/run.py**: Launch the web app.
- **data/process_data.py**: Execute the ETL Pipeline and generate a database with clean data.
- **data/disaster_categories.csv**: Table containing the categories asociated to the messages.
- **data/disaster_messages.csv**: Table containing the messages.
- **models/train_classifier.py**: Execute the ML Pipeline and generate the model in a pkl file.
