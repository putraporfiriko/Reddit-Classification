# we're on to black magic territory here

import os
import json
import string
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import collections
import re
import argparse
import sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# List all clean JSON files
clean_dir = './Successful JSON exports/clean'
json_files = [f for f in os.listdir(clean_dir)
              if f.endswith('.json') and os.path.isfile(os.path.join(clean_dir, f))]

# Display the list of JSON files for user selection
print("Available clean JSON files:")
for i, file in enumerate(json_files, 1):
    print(f"{i}. {file}")

# Let user choose a file
while True:
    try:
        choice = int(input("\nEnter the number of the file you want to load: "))
        if 1 <= choice <= len(json_files):
            selected_file = json_files[choice-1]
            file_path = os.path.join(clean_dir, selected_file)

            # Load the selected JSON file
            with open(file_path, 'r') as f:
                data_clean = json.load(f)

            print(f"Successfully loaded: {selected_file}")
            break
        else:
            print(f"Please enter a number between 1 and {len(json_files)}")
    except ValueError:
        print("Please enter a valid number")

# Convert the loaded JSON data to a pandas DataFrame
data_clean = pd.DataFrame(data_clean)

# Display basic information about the DataFrame


# Add command-line argument parsing
parser = argparse.ArgumentParser(description='Process JSON data with optional verbose output')
parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
parser.add_argument('-c', '--cosine-similarity', action='store_true', help='Display cosine similarity matrix')
args = parser.parse_args()

# Only display detailed DataFrame information if verbose flag is set
if args.verbose:
    print(f"\nDataFrame shape: {data_clean.shape}")
    print("\nFirst few rows:")
    print(data_clean.head())

data_clean = data_clean.astype({'postflairs' : 'category'})
data_clean = data_clean.astype({'posttitle' : 'string'})

# tf-idf. scary
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data_clean['posttitle'].astype('U'))

tf = TfidfVectorizer()
text_tf = tf.fit_transform(data_clean['posttitle'].astype('U'))
print(text_tf)

cos_sim=cosine_similarity(text_tf, text_tf)

if args.cosine_similarity:
    print(cos_sim)
#------------------------
#
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cos_sim, data_clean['postflairs'], test_size=0.2, random_state=33)
print("Jumlah Data Uji:", X_test.shape)
print("Jumlah Data Latih:",X_train.shape)

# Calculate counts for each category in test set
serious_test = (y_test == 'Serious').sum()
other_test = (y_test == 'Other').sum()
meme_test = (y_test == 'Meme').sum()
relationship_test = (y_test == 'Relationship').sum()
social_test = (y_test == 'Social').sum()
discussion_test = (y_test == 'Discussion').sum()
media_test = (y_test == 'Media').sum()
art_test = (y_test == 'Art').sum()
advice_test = (y_test == 'Advice').sum()
rant_test = (y_test == 'Rant').sum()
rip_wii_shop_test = (y_test == 'RIP Wii Shop').sum()
mod_test = (y_test == 'Mod').sum()
selfie_test = (y_test == 'Selfie').sum()

# Calculate counts for each category in training set
serious_train = (y_train == 'Serious').sum()
other_train = (y_train == 'Other').sum()
meme_train = (y_train == 'Meme').sum()
relationship_train = (y_train == 'Relationship').sum()
social_train = (y_train == 'Social').sum()
discussion_train = (y_train == 'Discussion').sum()
media_train = (y_train == 'Media').sum()
art_train = (y_train == 'Art').sum()
advice_train = (y_train == 'Advice').sum()
rant_train = (y_train == 'Rant').sum()
rip_wii_shop_train = (y_train == 'RIP Wii Shop').sum()
mod_train = (y_train == 'Mod').sum()
selfie_train = (y_train == 'Selfie').sum()
# Print counts for test data categories
print("Jumlah data uji Serious:", serious_test)
print("Jumlah data uji Other:", other_test)
print("Jumlah data uji Meme:", meme_test)
print("Jumlah data uji Relationship:", relationship_test)
print("Jumlah data uji Social:", social_test)
print("Jumlah data uji Discussion:", discussion_test)
print("Jumlah data uji Media:", media_test)
print("Jumlah data uji Art:", art_test)
print("Jumlah data uji Advice:", advice_test)
print("Jumlah data uji Rant:", rant_test)
print("Jumlah data uji RIP Wii Shop:", rip_wii_shop_test)
print("Jumlah data uji Mod:", mod_test)
print("Jumlah data uji Selfie:", selfie_test)

# Print counts for training data categories
print("Jumlah data latih Serious:", serious_train)
print("Jumlah data latih Other:", other_train)
print("Jumlah data latih Meme:", meme_train)
print("Jumlah data latih Relationship:", relationship_train)
print("Jumlah data latih Social:", social_train)
print("Jumlah data latih Discussion:", discussion_train)
print("Jumlah data latih Media:", media_train)
print("Jumlah data latih Art:", art_train)
print("Jumlah data latih Advice:", advice_train)
print("Jumlah data latih Rant:", rant_train)
print("Jumlah data latih RIP Wii Shop:", rip_wii_shop_train)
print("Jumlah data latih Mod:", mod_train)
print("Jumlah data latih Selfie:", selfie_train)
data_clean['postflairs'].value_counts()

#------------------------
# implementing KNN
# perform algoritma KNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
predicted = clf.predict(X_test)
print(f'confusion matrix:\n {confusion_matrix(y_test, predicted)}')
print('===============================================\n')
# Get the confusion matrix
cm = confusion_matrix(y_test, predicted)

# Print confusion matrix for each category
categories = ['Serious', 'Other', 'Meme', 'Relationship', 'Social', 'Discussion',
              'Media', 'Art', 'Advice', 'Rant', 'RIP Wii Shop', 'Mod', 'Selfie']

print("Confusion Matrix by Category:")
for i, category in enumerate(categories):
    if category in y_train.unique():
        # For each category, calculate TP, FP, FN, TN
        # Check if category index is within confusion matrix bounds
        idx = list(y_test.unique()).index(category) if category in y_test.unique() else -1
        if idx >= 0 and idx < len(cm):
            true_positives = cm[idx, idx]
            false_positives = cm[:, idx].sum() - true_positives
            false_negatives = cm[idx, :].sum() - true_positives
            true_negatives = cm.sum() - (true_positives + false_positives + false_negatives)
        else:
            # Handle the case where the category isn't in the test set
            true_positives = false_positives = false_negatives = 0
            true_negatives = cm.sum()

        print(f"\n{category}:")
        print(f"TP: {true_positives} (correctly predicted as {category})")
        print(f"FP: {false_positives} (incorrectly predicted as {category})")
        print(f"FN: {false_negatives} (incorrectly predicted as not {category})")
        print(f"TN: {true_negatives} (correctly predicted as not {category})")
print(classification_report(y_test, predicted, zero_division=0))
print('===============================================\n')
# Extract subreddit name from the filename
subreddit = selected_file.split('_')[0]
print(f"Classification results for r/{subreddit}:")
print("Accuracy:" , accuracy_score(y_test,predicted))
print("Precision:" , precision_score(y_test, predicted, average="weighted"))
print("Recall:" , recall_score(y_test, predicted, average="weighted"))
print("f1_score:" , f1_score(y_test, predicted, average="weighted"))
print("error_rate:", 1-accuracy_score(y_test,predicted))


#------------------------
# implementing Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
rf_predicted = rf_clf.predict(X_test)
print(f'confusion matrix:\n {confusion_matrix(y_test, rf_predicted)}')
print('===============================================\n')
rf_cm = confusion_matrix(y_test, rf_predicted)
categories = ['Serious', 'Other', 'Meme', 'Relationship', 'Social', 'Discussion',
              'Media', 'Art', 'Advice', 'Rant', 'RIP Wii Shop', 'Mod', 'Selfie']

print("Confusion Matrix by Category (Random Forest):")
for i, category in enumerate(categories):
    if category in y_train.unique():
        idx = list(y_test.unique()).index(category) if category in y_test.unique() else -1
        if idx >= 0 and idx < len(rf_cm):
            true_positives = rf_cm[idx, idx]
            false_positives = rf_cm[:, idx].sum() - true_positives
            false_negatives = rf_cm[idx, :].sum() - true_positives
            true_negatives = rf_cm.sum() - (true_positives + false_positives + false_negatives)
        else:
            true_positives = false_positives = false_negatives = 0
            true_negatives = rf_cm.sum()
        print(f"\n{category}:")
        print(f"TP: {true_positives} (correctly predicted as {category})")
        print(f"FP: {false_positives} (incorrectly predicted as {category})")
        print(f"FN: {false_negatives} (incorrectly predicted as not {category})")
        print(f"TN: {true_negatives} (correctly predicted as not {category})")
print(classification_report(y_test, rf_predicted, zero_division=0))
print('===============================================\n')
subreddit = selected_file.split('_')[0]
print(f"Random Forest classification results for r/{subreddit}:")
print("Accuracy:" , accuracy_score(y_test, rf_predicted))
print("Precision:" , precision_score(y_test, rf_predicted, average="weighted"))
print("Recall:" , recall_score(y_test, rf_predicted, average="weighted"))
print("f1_score:" , f1_score(y_test, rf_predicted, average="weighted"))
print("error_rate:", 1-accuracy_score(y_test, rf_predicted))

#------------------------
# implementing Gradient Boosting
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_clf.fit(X_train, y_train)
gb_predicted = gb_clf.predict(X_test)
print(f'confusion matrix:\n {confusion_matrix(y_test, gb_predicted)}')
print('===============================================\n')
gb_cm = confusion_matrix(y_test, gb_predicted)
print("Confusion Matrix by Category (Gradient Boosting):")
for i, category in enumerate(categories):
    if category in y_train.unique():
        idx = list(y_test.unique()).index(category) if category in y_test.unique() else -1
        if idx >= 0 and idx < len(gb_cm):
            true_positives = gb_cm[idx, idx]
            false_positives = gb_cm[:, idx].sum() - true_positives
            false_negatives = gb_cm[idx, :].sum() - true_positives
            true_negatives = gb_cm.sum() - (true_positives + false_positives + false_negatives)
        else:
            true_positives = false_positives = false_negatives = 0
            true_negatives = gb_cm.sum()
        print(f"\n{category}:")
        print(f"TP: {true_positives} (correctly predicted as {category})")
        print(f"FP: {false_positives} (incorrectly predicted as {category})")
        print(f"FN: {false_negatives} (incorrectly predicted as not {category})")
        print(f"TN: {true_negatives} (correctly predicted as not {category})")
print(classification_report(y_test, gb_predicted, zero_division=0))
print('===============================================\n')
print(f"Gradient Boosting classification results for r/{subreddit}:")
print("Accuracy:" , accuracy_score(y_test, gb_predicted))
print("Precision:" , precision_score(y_test, gb_predicted, average="weighted"))
print("Recall:" , recall_score(y_test, gb_predicted, average="weighted"))
print("f1_score:" , f1_score(y_test, gb_predicted, average="weighted"))
print("error_rate:", 1-accuracy_score(y_test, gb_predicted))

#------------------------
# implementing XGBoost
xg_clf = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
# Encode y_train and y_test to integer labels for XGBoost (retarded hotpatch 1. fuck you xgboost)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

xg_clf.fit(X_train, y_train_encoded)
xg_predicted = xg_clf.predict(X_test)
# Convert xg_predicted (which is integer labels) back to original string labels for comparison (retarded hotpatch 2)
xg_predicted_labels = le.inverse_transform(xg_predicted)
print(f'confusion matrix:\n {confusion_matrix(y_test, xg_predicted_labels)}')
print('===============================================\n')
xg_cm = confusion_matrix(y_test, xg_predicted_labels)
print("Confusion Matrix by Category (XGBoost):")
for i, category in enumerate(categories):
    if category in y_train.unique():
        idx = list(y_test.unique()).index(category) if category in y_test.unique() else -1
        if idx >= 0 and idx < len(xg_cm):
            true_positives = xg_cm[idx, idx]
            false_positives = xg_cm[:, idx].sum() - true_positives
            false_negatives = xg_cm[idx, :].sum() - true_positives
            true_negatives = xg_cm.sum() - (true_positives + false_positives + false_negatives)
        else:
            true_positives = false_positives = false_negatives = 0
            true_negatives = xg_cm.sum()
        print(f"\n{category}:")
        print(f"TP: {true_positives} (correctly predicted as {category})")
        print(f"FP: {false_positives} (incorrectly predicted as {category})")
        print(f"FN: {false_negatives} (incorrectly predicted as not {category})")
        print(f"TN: {true_negatives} (correctly predicted as not {category})")
print(classification_report(y_test, xg_predicted_labels, zero_division=0))
print('===============================================\n')
print(f"XGBoost classification results for r/{subreddit}:")
print("Accuracy:" , accuracy_score(y_test, xg_predicted_labels))
print("Precision:" , precision_score(y_test, xg_predicted_labels, average="weighted"))
print("Recall:" , recall_score(y_test, xg_predicted_labels, average="weighted"))
print("f1_score:" , f1_score(y_test, xg_predicted_labels, average="weighted"))
print("error_rate:", 1-accuracy_score(y_test, xg_predicted_labels))

#------------------------
# implementing AdaBoost
adb_clf = AdaBoostClassifier(n_estimators=100, random_state=42)
adb_clf.fit(X_train, y_train)
adb_predicted = adb_clf.predict(X_test)
print(f'confusion matrix:\n {confusion_matrix(y_test, adb_predicted)}')
print('===============================================\n')
adb_cm = confusion_matrix(y_test, adb_predicted)
print("Confusion Matrix by Category (AdaBoost):")
for i, category in enumerate(categories):
    if category in y_train.unique():
        idx = list(y_test.unique()).index(category) if category in y_test.unique() else -1
        if idx >= 0 and idx < len(adb_cm):
            true_positives = adb_cm[idx, idx]
            false_positives = adb_cm[:, idx].sum() - true_positives
            false_negatives = adb_cm[idx, :].sum() - true_positives
            true_negatives = adb_cm.sum() - (true_positives + false_positives + false_negatives)
        else:
            true_positives = false_positives = false_negatives = 0
            true_negatives = adb_cm.sum()
        print(f"\n{category}:")
        print(f"TP: {true_positives} (correctly predicted as {category})")
        print(f"FP: {false_positives} (incorrectly predicted as {category})")
        print(f"FN: {false_negatives} (incorrectly predicted as not {category})")
        print(f"TN: {true_negatives} (correctly predicted as not {category})")
print(classification_report(y_test, adb_predicted, zero_division=0))
print('===============================================\n')
print(f"AdaBoost classification results for r/{subreddit}:")
print("Accuracy:" , accuracy_score(y_test, adb_predicted))
print("Precision:" , precision_score(y_test, adb_predicted, average="weighted"))
print("Recall:" , recall_score(y_test, adb_predicted, average="weighted"))
print("f1_score:" , f1_score(y_test, adb_predicted, average="weighted"))
print("error_rate:", 1-accuracy_score(y_test, adb_predicted))

# visualize the scores using matplotlib


# Collect metrics for each model
model_names = ['KNN', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'AdaBoost']
accuracies = [
    accuracy_score(y_test, predicted),
    accuracy_score(y_test, rf_predicted),
    accuracy_score(y_test, gb_predicted),
    accuracy_score(y_test, xg_predicted_labels),
    accuracy_score(y_test, adb_predicted)
]
precisions = [
    precision_score(y_test, predicted, average="weighted", zero_division=0),
    precision_score(y_test, rf_predicted, average="weighted", zero_division=0),
    precision_score(y_test, gb_predicted, average="weighted", zero_division=0),
    precision_score(y_test, xg_predicted_labels, average="weighted", zero_division=0),
    precision_score(y_test, adb_predicted, average="weighted", zero_division=0)
]
recalls = [
    recall_score(y_test, predicted, average="weighted", zero_division=0),
    recall_score(y_test, rf_predicted, average="weighted", zero_division=0),
    recall_score(y_test, gb_predicted, average="weighted", zero_division=0),
    recall_score(y_test, xg_predicted_labels, average="weighted", zero_division=0),
    recall_score(y_test, adb_predicted, average="weighted", zero_division=0)
]
f1_scores = [
    f1_score(y_test, predicted, average="weighted", zero_division=0),
    f1_score(y_test, rf_predicted, average="weighted", zero_division=0),
    f1_score(y_test, gb_predicted, average="weighted", zero_division=0),
    f1_score(y_test, xg_predicted_labels, average="weighted", zero_division=0),
    f1_score(y_test, adb_predicted, average="weighted", zero_division=0)
]
error_rates = [1 - acc for acc in accuracies]

# Convert to percent and round to 3 decimals
def fmt(x):
    return f"{x*100:.1f}%"

x = np.arange(len(model_names))
width = 0.15

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - 2*width, accuracies, width, label='Accuracy')
rects2 = ax.bar(x - width, precisions, width, label='Precision')
rects3 = ax.bar(x, recalls, width, label='Recall')
rects4 = ax.bar(x + width, f1_scores, width, label='F1 Score')
rects5 = ax.bar(x + 2*width, error_rates, width, label='Error Rate')

# Add value labels
for rects, vals in zip([rects1, rects2, rects3, rects4, rects5], [accuracies, precisions, recalls, f1_scores, error_rates]):
    for rect, val in zip(rects, vals):
        height = rect.get_height()
        ax.annotate(fmt(val),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

ax.set_ylabel('Score')
ax.set_title('Model Performance Metrics')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylim(0, 1.1)
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()

#------------------------
# Prepare data for model training
# Since cos_sim is a similarity matrix, but classifiers expect feature vectors,
# we'll use the TF-IDF features (text_tf) for these models.
# Also, ensure y_train and y_test are properly formatted

# Convert sparse matrix to dense for scaling
X_train_dense = X_train if not hasattr(X_train, "toarray") else X_train.toarray()
X_test_dense = X_test if not hasattr(X_test, "toarray") else X_test.toarray()

# Standardize features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train_dense)
x_test_scaled = scaler.transform(X_test_dense)
y_train_scaled = y_train
y_test_scaled = y_test

# Define parameter grids for each model
rf_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
gb_param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.1, 0.05],
    'max_depth': [3, 5]
}
xg_param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.1, 0.05],
    'max_depth': [3, 5]
}
adb_param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [1.0, 0.5]
}

# Create and fit GridSearchCV objects for each model
models = {
    'rf': (RandomForestClassifier(), rf_param_grid),
    'gb': (GradientBoostingClassifier(), gb_param_grid),
    #'xg': (xgb.XGBClassifier(eval_metric='mlogloss'), xg_param_grid),
    'adb': (AdaBoostClassifier(), adb_param_grid),
}

best_estimators = {}

for name, (model, param_grid) in models.items():
    print(f"\nRunning GridSearchCV for {name}...")
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    # For XGBoost, encode y labels as integers (retarded hotpatch 3)
    if name == 'xg':
        y_train_enc = le.transform(y_train_scaled)
        grid_search.fit(x_train_scaled, y_train_enc)
    else:
        grid_search.fit(x_train_scaled, y_train_scaled)
    print(f"Best hyperparameters for {name}: {grid_search.best_params_}")
    best_estimators[name] = grid_search.best_estimator_

    # Evaluate on test set
    y_pred = grid_search.predict(x_test_scaled)
    print(f"Results for {name}:")
    print(f"Accuracy: {accuracy_score(y_test_scaled, y_pred)}")
    # For XGBoost, y_pred will be integer-encoded, so decode for metrics
    if name == 'xg':
        y_pred_labels = le.inverse_transform(y_pred)
        print(f"Precision: {precision_score(y_test_scaled, y_pred_labels, average='weighted', zero_division=0)}")
    else:
        print(f"Precision: {precision_score(y_test_scaled, y_pred, average='weighted', zero_division=0)}")
    print(f"Recall: {recall_score(y_test_scaled, y_pred, average='weighted', zero_division=0)}")
    print(f"F1 Score: {f1_score(y_test_scaled, y_pred, average='weighted', zero_division=0)}")
    print("Classification Report:")
    print(classification_report(y_test_scaled, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_scaled, y_pred))
