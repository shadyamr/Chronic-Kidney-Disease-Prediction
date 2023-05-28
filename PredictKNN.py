import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv("C:/Users/shady/Downloads/ai-project/Preprocessed.csv")

# Create a list of non-numeric columns
non_numeric_columns = df.select_dtypes(exclude="number").columns

# Create a dictionary to store the label encoders
label_encoders = {}

# Iterate over the non-numeric columns
for column in non_numeric_columns:
    # Create a label encoder for the column
    encoder = LabelEncoder()
    
    # Fit the encoder to the column
    encoder.fit(df[column])
    
    # Transform the column using the encoder
    df[column] = encoder.transform(df[column])
    
    # Store the encoder in the dictionary
    label_encoders[column] = encoder

# Impute missing values using the mean strategy
imputer = SimpleImputer(strategy="mean")
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Split the data into a training set and a test set
X = df.drop("class", axis=1)
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create the KNN model
knn = KNeighborsClassifier()

# Fit the model to the training data
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print(f"Accuracy: {accuracy:.2f}")