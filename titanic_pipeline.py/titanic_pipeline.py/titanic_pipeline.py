
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(train_path, test_path):
    """
    Load and preprocess the Titanic dataset.
    """
   
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    
    test_df['Survived'] = np.nan
    full_df = pd.concat([train_df, test_df], sort=False)

    
    full_df['Age'].fillna(full_df['Age'].median(), inplace=True)
    full_df['Embarked'].fillna(full_df['Embarked'].mode()[0], inplace=True)
    full_df['Fare'].fillna(full_df['Fare'].median(), inplace=True)

    
    full_df['FamilySize'] = full_df['SibSp'] + full_df['Parch'] + 1
    full_df['Title'] = full_df['Name'].str.extract(r',\s*([^\.]*)\.')[0]

    
    full_df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

    
    le = LabelEncoder()
    full_df['Sex'] = le.fit_transform(full_df['Sex'])
    full_df = pd.get_dummies(full_df, columns=['Embarked', 'Title'], drop_first=True)

    
    train_data = full_df[~full_df['Survived'].isna()]
    test_data = full_df[full_df['Survived'].isna()].drop(['Survived'], axis=1)

    return train_data, test_data

def build_pipeline():
    """
    Build a machine learning pipeline with a RandomForestClassifier and StandardScaler.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(random_state=42))
    ])

def tune_hyperparameters(pipeline, X_train, y_train):
    """
    Perform hyperparameter tuning using GridSearchCV.
    """
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def main():
   
    train_path = 'train.csv'
    test_path = 'test.csv'

    
    train_data, test_data = preprocess_data(train_path, test_path)

    
    X = train_data.drop(['Survived', 'PassengerId'], axis=1)
    y = train_data['Survived'].astype(int)
    X_test_final = test_data.drop('PassengerId', axis=1)
    test_ids = test_data['PassengerId']

   
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    
    pipeline = build_pipeline()
    best_model = tune_hyperparameters(pipeline, X_train, y_train)

    
    val_preds = best_model.predict(X_val)
    accuracy = accuracy_score(y_val, val_preds)
    print(f'Validation Accuracy: {accuracy:.4f}')
    print("Classification Report:")
    print(classification_report(y_val, val_preds))

    cv_scores = cross_val_score(best_model, X, y, cv=5)
    print(f'Cross-Validation Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

    test_preds = best_model.predict(X_test_final)
    submission = pd.DataFrame({'PassengerId': test_ids, 'Survived': test_preds.astype(int)})
    submission.to_csv('submission.csv', index=False)
    print("✅ Submission saved as 'submission.csv'")

if __name__ == "__main__":
    main()