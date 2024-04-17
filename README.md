# Wine Quality Classification

## Setup Instructions:

I. **Install Dependencies:**
   - Ensure Python is installed on your system.
   - Run the following commands in a terminal or command prompt:

     ```bash
     pip install pandas numpy scikit-learn ucimlrepo imbalanced-learn
     ```

II. **Installing and Preparing the Information:**
1. A URL is used to load the dataset. Make sure you are connected to the internet.
2. To load the dataset, import pandas:
  import pandas as pd
3. Use pd.read_csv() to load the dataset by providing the correct URL.
4. Examine the dataset and carry out any required cleanup.

III. **Follow Code Steps:**
Training and Test Split:
1. To evaluate the model's performance on unobserved data, divide the dataset into training and test sets.
2. To guarantee a representative sample for model assessment, data was randomly assigned to training and test sets using the train_test_split function.

Feature Scaling:

1. In order to ensure consistent ranges for accurate model training, normalize the numerical characteristics to a standard scale.
2. Scaled training and test datasets using scikit-learn's StandardScaler.
 
Model Training:
1. Investigated and assessed the efficacy of two categorization models: decision trees and logistic regression.
2. For a strong model evaluation, repeated K-Fold Cross-Validation (k=10) was used.

Model Evaluation:
1. Models were assessed using important metrics including F1-score, accuracy, precision, and recall.
2. To offer comprehensive insights into model performance, a classification report and confusion matrix were generated.

Balancing the Dataset: 
1. Class imbalance in the training set was addressed by oversampling the minority class using the imbalanced-learn package.
2. To improve model training on minority class data, RandomOverSampler was applied to balance the class distribution.

Final Model Training and Evaluation:
1. Using the results of cross-validation, the best-performing model (Decision Tree or Logistic Regression) was determined.
2. To strengthen the model's capacity to manage uneven classes, it was trained on the oversampled training set.

## Additional Notes:

- Stable internet connection is required for fetching the dataset from UCI ML Repository.
- Refer to comments in the code for assistance.
- Code must be executed in an order, which is present in the python file.
