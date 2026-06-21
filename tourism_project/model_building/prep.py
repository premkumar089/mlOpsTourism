
'''
1. Prepare for data manipulation
2. Initialize the API by using the HF token. This will establish the connection with HF.
3. DATASET_PATH = "hf://datasets/<-------Hugging Face user ID --------->/projectname/file.csv"
4. Read the csv file
5. Seperate the target variable and independent variables
6. Seperate the categorical and numerical variables of Indepenent variables.
7. Split the dataset into train test using train test split function
8. train and test data - load this data into respective csv files
  "Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"
9. upload these csv into hugging face respository under
  repo_id="<-------Hugging Face user ID --------->/yourproject",
  repo_type= dataset


'''
