import json
from collections import Counter

def count_flairs_in_file(file_path):
    try:
        # Open and read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Extract the postflairs from each entry
        flairs = []
        if isinstance(data, list):
            for entry in data:
                if 'postflairs' in entry:
                    # Check if postflairs is a list
                    if isinstance(entry['postflairs'], list):
                        for flair in entry['postflairs']:
                            flairs.append(flair)
                    else:
                        flairs.append(entry['postflairs'])
        elif isinstance(data, dict):
            if 'postflairs' in data:
                # Check if postflairs is a list
                if isinstance(data['postflairs'], list):
                    for flair in data['postflairs']:
                        flairs.append(flair)
                else:
                    flairs.append(data['postflairs'])

        # Count occurrences of each flair
        flair_counter = Counter(flairs)

        # Get unique flairs and their counts
        unique_flairs = list(flair_counter.keys())

        # Print the results
        print("Unique postflairs:")
        for i, flair in enumerate(unique_flairs, 1):
            print(f"{i}. {flair}")

        print("\nPostflair counts:")
        for flair, count in flair_counter.items():
            print(f"{flair}: {count}")

        return flair_counter, unique_flairs

    except Exception as e:
        print(f"Error processing file: {e}")
        return None, None

# Example usage
import os

# Get all JSON files in the specified directory (not including subfolders)
json_files = [f for f in os.listdir('./Successful JSON exports/clean')
              if f.endswith('.json') and os.path.isfile(os.path.join('./Successful JSON exports/clean', f))]

# Display the list of JSON files for user selection
print("Available JSON files:")
for i, file in enumerate(json_files, 1):
    print(f"{i}. {file}")

# Let user choose a file
while True:
    try:
        choice = int(input("\nEnter the number of the file you want to analyze: "))
        if 1 <= choice <= len(json_files):
            selected_file = json_files[choice-1]
            file_path = os.path.join('./Successful JSON exports/clean', selected_file)

            # Count flairs in the selected file
            count_flairs_in_file(file_path)
            break
        else:
            print(f"Please enter a number between 1 and {len(json_files)}")
    except ValueError:
        print("Please enter a valid number")
