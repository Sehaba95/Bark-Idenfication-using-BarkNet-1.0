import os 
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# Get dataset path
script_path = os.path.realpath(__file__)
barknet_dataset_path = "/".join(script_path.split("/")[:-1]) + "/" 

# Load BarkNet Lookup 
data = pd.read_csv('lookup.csv')

# Convert Lookup to dictionary 
bark_dict = {}
for row in data.iterrows():
	class_id = row[1]['Class ID']
	folder_name = row[1]['Folder name']
	class_name = row[1]['Class name']
	bark_dict[folder_name] = [class_id, folder_name, class_name]

################################################################################################
# BarkNet 1.0 CSV generator
bark_csv = []

# Iterate over every dataset folder
for _, dirs, _ in os.walk(barknet_dataset_path):
	# Iterate over folders
	for directory in dirs:
		class_folder = os.path.join(barknet_dataset_path, directory)
		# Get images of every class folder
		for _, _, data_samples in os.walk(class_folder):
			for data_sample in data_samples:
				img_path = "/".join([directory, data_sample])
				classname = bark_dict[directory]
				# ['Image path', 'Directory', 'Class ID', 'Class name']
				bark_csv.append([img_path, directory, classname[0], classname[2]]) # 0: class_ID 2: class_name

print("Number of data samples: ", len(bark_csv))

bark_csv = pd.DataFrame(bark_csv, columns=['Image path', 'Directory', 'Class ID', 'Class name'])
bark_csv.to_csv("BarkNet_1_0.csv")
print(bark_csv.head())
print(bark_csv.columns)

################################################################################################
# Split the BarkNet 1.0 into train, validation and test sets 

X = bark_csv[['Image path', 'Directory', 'Class name']]
y = bark_csv[['Class ID']]

# sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

y_train = pd.DataFrame(y_train)
print("Training data")
print("Number of training data samples: {}".format(len(y_train)))
print("Number of classes available in the training data: {}\n".format(len(dict(y_train['Class ID'].value_counts()).keys())))

y_val = pd.DataFrame(y_val)
print("Validation data")
print("Number of validation data samples: {}".format(len(y_val)))
print("Number of classes available in the validation data: {}\n".format(len(dict(y_val['Class ID'].value_counts()).keys())))

y_test = pd.DataFrame(y_test)
print("Testing data")
print("Number of testing data samples: {}".format(len(y_test)))
print("Number of classes available in the testing data: {}\n".format(len(dict(y_test['Class ID'].value_counts()).keys())))


bark_train = pd.DataFrame(X_train)
bark_train["Class ID"] = y_train
bark_train = bark_train[['Image path', 'Directory', 'Class ID', 'Class name']]
bark_train.to_csv("barknet_train.csv")

bark_test = pd.DataFrame(X_test)
bark_test["Class ID"] = y_test
bark_test = bark_test[['Image path', 'Directory', 'Class ID', 'Class name']]
bark_test.to_csv("barknet_test.csv")

bark_test = pd.DataFrame(X_val)
bark_test["Class ID"] = y_val
bark_test = bark_test[['Image path', 'Directory', 'Class ID', 'Class name']]
bark_test.to_csv("barknet_validation.csv")






















