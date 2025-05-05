#!/usr/bin/env python
import os
import joblib
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def create_sample_model(output_path="./model.joblib", use_pickle=False):
    """
    Create and save a sample machine learning model.
    
    Args:
        output_path: Path where to save the model
        use_pickle: Whether to use pickle instead of joblib
    """
    print(f"Creating sample model at {output_path}")
    
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
  
    print("Training a RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
  
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
   
    if use_pickle:
        print(f"Saving model using pickle with protocol 4...")
        with open(output_path, 'wb') as f:
            pickle.dump(model, f, protocol=4)
    else:
        print(f"Saving model using joblib with protocol 4...")
        joblib.dump(model, output_path, protocol=4)
    
    print(f"Model saved successfully to {output_path}")
    
    print("\nSample prediction input (single sample):")
    sample = X_test[0].tolist()
    print(f"{{\"data\": {sample}}}")
    
    print("\nSample prediction input (multiple samples):")
    samples = X_test[:3].tolist()
    print(f"{{\"data\": {samples}}}")
    
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create a sample ML model")
    parser.add_argument(
        "--output", 
        type=str, 
        default="./model.joblib",
        help="Path where to save the model"
    )
    parser.add_argument(
        "--pickle", 
        action="store_true",
        help="Use pickle instead of joblib"
    )
    args = parser.parse_args()   
    create_sample_model(args.output, args.pickle)