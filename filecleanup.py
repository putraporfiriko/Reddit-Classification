# We're gonna do some data cleanup on this script.
# This mainly revolves around case folding (in retard's term: data(lower))
# Cleaning up punctuations and anything else using regex
# And tokenizing the data.
# As a bonus, we will normalize the data, and remove stopwords.

import os
import json
import codecs
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# step 1: load the data.
# Get all JSON files in the specified directory (not including subfolders)
json_files = [f for f in os.listdir('./Successful JSON exports')
              if f.endswith('.json') and os.path.isfile(os.path.join('./Successful JSON exports', f))]

# Display the list of JSON files for user selection
print("Available JSON files:")
for i, file in enumerate(json_files, 1):
    print(f"{i}. {file}")

# Let user choose a file
while True:
    try:
        choice = int(input("\nEnter the number of the file you want to load: "))
        if 1 <= choice <= len(json_files):
            selected_file = json_files[choice-1]
            file_path = os.path.join('./Successful JSON exports', selected_file)

            # Load the selected JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)

            print(f"Successfully loaded: {selected_file}")
            break
        else:
            print(f"Please enter a number between 1 and {len(json_files)}")
    except ValueError:
        print("Please enter a valid number")

# Remove dicts where postflairs is null or posttitle is present with postflairs as null
data = [item for item in data if not (item.get('postflairs') is None and 'posttitle' in item)]

# step 2: casefolding

# Create a buffer file for the cleaned data
buffer_file_path = 'cleaned_data_buffer.json'

# Case folding for post titles
for item in data:
    if 'posttitle' in item:
        item['posttitle'] = item['posttitle'].lower()

# Save the modified data to the buffer file
with codecs.open(buffer_file_path, 'w', 'utf-8') as buffer_file:
    json.dump(data, buffer_file, ensure_ascii=False, indent=4)

print(f"Casefolding completed. Data saved to buffer file: {buffer_file_path}")

# step 3: clean up symbols (again, only for posttitles)

# Cleaning symbols in post titles using regex
for item in data:
    if 'posttitle' in item:
        # Remove special characters, keeping only alphanumeric and whitespace
        item['posttitle'] = re.sub(r'[^\w\s]', '', item['posttitle'])
        # Replace multiple spaces with a single space
        item['posttitle'] = re.sub(r'\s+', ' ', item['posttitle']).strip()

# Save the modified data back to the buffer file
with codecs.open(buffer_file_path, 'w', 'utf-8') as buffer_file:
    json.dump(data, buffer_file, ensure_ascii=False, indent=4)

print(f"Symbol cleanup completed. Data saved to buffer file: {buffer_file_path}")

# step 4: tokenizing, normalization

# Download required NLTK data if not already available
try:
    # Check if the 'punkt' tokenizer data exists
    nltk.data.find('tokenizers/punkt')
    print("NLTK punkt tokenizer already downloaded.")
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

# step 5: remove stopwords

# Download stopwords if not already available
try:
    nltk.data.find('corpora/stopwords')
    print("NLTK stopwords already downloaded.")
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

# Get English stopwords
stop_words = set(stopwords.words('english'))

# Process text and remove stopwords without storing tokens
for item in data:
    if 'posttitle' in item:
        # Tokenize without storing
        tokens = word_tokenize(item['posttitle'])
        # Filter out stopwords and join back into a cleaned posttitle
        item['posttitle'] = ' '.join([word for word in tokens if word.lower() not in stop_words])

# Create the clean directory if it doesn't exist
clean_dir = './Successful JSON exports/clean/'
os.makedirs(clean_dir, exist_ok=True)

# Save the modified data to the cleaned file
cleaned_file_path = os.path.join(clean_dir, f"{os.path.splitext(selected_file)[0]}_clean.json")
with codecs.open(cleaned_file_path, 'w', 'utf-8') as clean_file:
    json.dump(data, clean_file, ensure_ascii=False, indent=4)

print(f"Stopword removal completed and post titles updated. Data saved to cleaned file: {cleaned_file_path}")
