import pandas as pd
import skops.io as sio
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Load the necessary libraries:
# - pandas for data manipulation
# - skops.io for saving and loading models
# - sklearn modules for various machine learning tasks like encoding, imputing, scaling, and model training

## Loading the Data
drug_df = pd.read_csv("./Data/drug.csv")  # Load the dataset from a CSV file
drug_df = drug_df.sample(frac=1)  # Shuffle the dataset to ensure random distribution of samples

## Train Test Split
from sklearn.model_selection import train_test_split

X = drug_df.drop("Drug", axis=1).values  # Feature matrix (all columns except 'Drug')
y = drug_df.Drug.values  # Target vector (only 'Drug' column)

# Split the dataset into training and testing sets with 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=125
)

## Pipeline Configuration
cat_col = [1, 2, 3]  # List of categorical columns (column indices 1, 2, and 3)
num_col = [0, 4]  # List of numerical columns (column indices 0 and 4)

# Create a ColumnTransformer to apply different preprocessing steps to different columns
transform = ColumnTransformer(
    [
        ("encoder", OrdinalEncoder(), cat_col),  # Apply OrdinalEncoder to the categorical columns
        ("num_imputer", SimpleImputer(strategy="median"), num_col),  # Handle missing values in numerical columns using median
        ("num_scaler", StandardScaler(), num_col),  # Scale numerical columns to have zero mean and unit variance
    ]
)

# Create a pipeline to streamline the entire process (preprocessing + model training)
pipe = Pipeline(
    steps=[
        ("preprocessing", transform),  # First step: apply the ColumnTransformer for preprocessing
        ("model", RandomForestClassifier(n_estimators=10, random_state=125)),  # Second step: fit a RandomForestClassifier model
    ]
)

## Model Training
pipe.fit(X_train, y_train)  # Train the pipeline (preprocessing + model) on the training data

## Model Evaluation
predictions = pipe.predict(X_test)  # Make predictions on the test data
accuracy = accuracy_score(y_test, predictions)  # Calculate accuracy score
f1 = f1_score(y_test, predictions, average="macro")  # Calculate F1 score (macro-averaged for multi-class classification)

# Print the evaluation metrics
print("Accuracy:", str(round(accuracy, 2) * 100) + "%", "F1:", round(f1, 2))

## Confusion Matrix Plot
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Generate the confusion matrix and plot it
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)  # Compute the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)  # Create a confusion matrix display
disp.plot()  # Plot the confusion matrix
plt.savefig("./Results/model_results.png", dpi=120)  # Save the plot as a PNG image in the 'Results' folder

## Write metrics to file
# Save accuracy and F1 score to a text file in the 'Results' folder
with open("./Results/metrics.txt", "w") as outfile:
    outfile.write(f"\nAccuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}")

## Saving the model file
# Save the entire trained pipeline (including the preprocessing steps) using skops to a file for later use
sio.dump(pipe, "./Model/drug_pipeline.skops")