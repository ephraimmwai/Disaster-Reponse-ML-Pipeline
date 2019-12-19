# Disaster Response Pipeline Project
![Message Classifier](https://github.com/ephraimmwai/Disaster-Reponse-ML-Pipeline/blob/master/static/vendor/img/Capture.PNG)

### Instructions:
1. Clone the [repository](https://github.com/ephraimmwai/Disaster-Reponse-ML-Pipeline)

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores clean text meassages in a database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run the Machine Learning pipeline that trains a classifier and saves the model;
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        OR 
         Download the already trained [classifier](https://drive.google.com/file/d/1rBphtBMF3uQNrfz31es0k8AdR7AAvbOL/view) (900 MB ) and save it inside the ```models``` folder

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://127.0.0.1:3001/ on your web browser
