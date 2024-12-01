# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

# Requirements
- A requirements.txt file is included in the repo to allow pip installation of the required libaries. Please open that file to see the specific requirements and additional information.

# Project Overview / CRISP-DM Flow

## Business Understanding
- The purpose of this project is to identify an efficient way to model emergency messages to determine which are most informative for making a quick disaster response decision. This is important as resources for disaster response are limited and false positives can prevent those resources from being deployed where they are needed most.

## Data Understanding
- The dataset is raw, containing an identifier along with communication information that can contain an English 'message', a non-English 'original' messge, and a 'genre' (type of communication). This insformation is provided in CSV form.
- The content of each row is inconsistent - for example, some of the lines contain more than 4 commas (which would cleanly correspond to the 4 expected fields), some contain consecutive commas (where one of the messages is missing), and some contain quotation marks separating the English and non-English messages.
- There is also categorical information stored in a separate CSV, which provides some informative identifiers that are pre-associated with each message (and link by an 'id' field in each set of data).
    - This CSV data is further subdivided into key/value pairs separated by a semicolon, e.g. 'id,key1-value1;key2-value2; ...' etc.
    - There are 36 categeries in the file, and the rubric says these are to be used as the ***responses*** for the multi-output classification task.
    - Note that the values in the category values are integers - mostly boolean (0 or 1) except for **related** which is ternery (0/1/2). More on this below.
    - These categories are also **not** mutually exclusive (the cross-sectional sum of them can be greater than 1).

- **load_data.py** loads the raw message and category information, first splits out the 'id' and the 'categories' key/value pairs, then splits the key/value pairs into columns of data with integer data, and joins each dataset (on 'id') and passes that joined dataframe along for cleaning in the next phase.

## Data Preparation
- **clean_data.py** attempts to separate the Enlish and non-English components of the raw nessage, and stores the processed English-only message as a new field in the data, allowing it to be more easily processed in the machine learning pipeline.
- It also converts the 'genre' field into a set of dummies with boolean values (n-1 categories).

## Data Modeling
- The clean_data() function in process_data.py applies some cleaning to the data.
    - By default, it will make the **related** field binary (by replacing the 2s with 1).
    - It will check for duplicate rows, and drop any that are found.
    - It will check for columns with constant values, and by default print a warning to the console about them (they can also be dropped, optionally).
        - The **child_alone** category is a constant: the rubric states to fit the model on all 36 categories, so this is being left in place. But as a constant, it adds no information to the model.
- The cleaned data is then passed to a Random Forest classifier.
    - A Scitkit-Learn pipelie is defined for modeling:
        - First, the **message** strings are split into lemmatized tokens, all lower case. Stop words and non-English words are removed.
        - The tokens are converted into a TF-IDF matrix.
        - The TF-IDF matrix is then used as the X variables to train the classifer on the Y data (the boolean values in the 36 categories).
    - The rubric mentions the dataset is imbalanced with some labels having fewer observations than others.
        - As mentioned previously, **child_alone** has no observations.
        - The fields **offer/shops/tools/fire/hospitals/missing_people/aid_centers/clothing/security** all have less than 500 observations.
            - But the smallest - **offer** - still has 118 observations, which is not negligible.
        - However, there are 6119 observations that have a 0/False value across **all** categories.
            - While these are a substantial portion of all samples, these are dropped by default so that their presence doesn't diminish the ability of the model to identify categories that are true.
            - In other words, message content in those observations can pull ALL categories towards zero.
        - There are no observations that have a 1/True value for all observations.

## Result Evaluation
- From command line, you can inspect various statistical metrics for the model running on test data, per category and overall.
- You can also run the **display_output()** function to show basic aggregate states (F1, Accuracy, Precision)

## Deployment
- Run the web app.

# Acknowledgments
- MANY thank you's to [Rajat](https://knowledge.udacity.com/questions/510253) on the Udacity forums - his go.html/master/html files solved my issue of Plotly graphs not displaying in my web app.