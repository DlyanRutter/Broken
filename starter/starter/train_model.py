# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pickle

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from ml.model import compute_confusion_matrix
import logging

logging.basicConfig(filename='journal.log',
                    level=logging.INFO,
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')

# Add code to load in the data.
datapath = "../data/census.csv"
data = pd.read_csv(datapath)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=10)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
                                            train, 
                                            categorical_features=cat_features,
                                            label="salary", 
                                            training=True
                                            )

# Proces the test data with the process_data function.
# Set train flag = False - We use the encoding from the train set
X_test, y_test, encoder, lb = process_data(
                                            test, 
                                            categorical_features=cat_features, 
                                            label="salary", 
                                            training=False,
                                            encoder=encoder,
                                            lb=lb
                                            )

# Train and save a model.
model = train_model(X_train, y_train)

# evaluate trained model on test set
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

logging.info(f"Classification target labels: {list(lb.classes_)}")
logging.info(f"precision:{precision:.3f}, recall:{recall:.3f}, fbeta:{fbeta:.3f}")

cm = compute_confusion_matrix(y_test, preds, labels=list(lb.classes_))

logging.info(f"Confusion matrix:\n{cm}")

# save model  to disk in ./model folder
filepath = '../model/trained_model.pkl'
pickle.dump(model, open(filepath, 'wb'))
logging.info(f"Model saved to disk: {filepath}")

# load the model from disk
#model = pickle.load(open(filepath, 'rb'))
