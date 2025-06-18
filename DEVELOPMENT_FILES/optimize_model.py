"""
Model optimization script for better accuracy
"""

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from simple_classifier import SimpleDeviceClassifier

def optimize_model():
    print("🚀 Optimizing model for better accuracy...")
    
    # Load data with better parameters
    classifier = SimpleDeviceClassifier(
        img_size=(128, 128),  # Higher resolution
        frames_per_video=8    # More frames for better temporal info
    )
    
    print("📊 Loading dataset with enhanced parameters...")
    X, y = classifier.load_dataset('correctly worn', 'incorrectly worn')
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Try multiple models with hyperparameter tuning
    models = {}
    
    # 1. Optimized Random Forest
    print("\n🌲 Optimizing Random Forest...")
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    rf = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X_train_scaled, y_train)
    
    rf_pred = rf_grid.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    models['Random Forest'] = {
        'model': rf_grid.best_estimator_,
        'accuracy': rf_accuracy,
        'params': rf_grid.best_params_
    }
    
    print(f"✅ Random Forest - Best accuracy: {rf_accuracy:.3f}")
    print(f"   Best params: {rf_grid.best_params_}")
    
    # 2. Optimized SVM
    print("\n🎯 Optimizing SVM...")
    svm_params = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01],
        'kernel': ['rbf', 'poly'],
        'class_weight': ['balanced']
    }
    
    svm = SVC(probability=True, random_state=42)
    svm_grid = GridSearchCV(svm, svm_params, cv=3, scoring='accuracy', n_jobs=-1)
    svm_grid.fit(X_train_scaled, y_train)
    
    svm_pred = svm_grid.predict(X_test_scaled)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    models['SVM'] = {
        'model': svm_grid.best_estimator_,
        'accuracy': svm_accuracy,
        'params': svm_grid.best_params_
    }
    
    print(f"✅ SVM - Best accuracy: {svm_accuracy:.3f}")
    print(f"   Best params: {svm_grid.best_params_}")
    
    # 3. Ensemble approach
    print("\n🤝 Creating ensemble model...")
    from sklearn.ensemble import VotingClassifier
    
    ensemble = VotingClassifier([
        ('rf', rf_grid.best_estimator_),
        ('svm', svm_grid.best_estimator_)
    ], voting='soft')
    
    ensemble.fit(X_train_scaled, y_train)
    ensemble_pred = ensemble.predict(X_test_scaled)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    models['Ensemble'] = {
        'model': ensemble,
        'accuracy': ensemble_accuracy,
        'params': 'RF + SVM ensemble'
    }
    
    print(f"✅ Ensemble - Accuracy: {ensemble_accuracy:.3f}")
    
    # Find best model
    best_model_name = max(models.keys(), key=lambda k: models[k]['accuracy'])
    best_model = models[best_model_name]
    
    print(f"\n🏆 Best model: {best_model_name}")
    print(f"   Accuracy: {best_model['accuracy']:.3f}")
    print(f"   Parameters: {best_model['params']}")
    
    # Detailed evaluation of best model
    print(f"\n📊 Detailed evaluation of {best_model_name}:")
    best_pred = best_model['model'].predict(X_test_scaled)
    print(classification_report(y_test, best_pred, 
                              target_names=['incorrectly_worn', 'correctly_worn']))
    
    # Save optimized model
    optimized_model_data = {
        'model': best_model['model'],
        'scaler': scaler,
        'img_size': (128, 128),
        'frames_per_video': 8,
        'model_type': best_model_name,
        'accuracy': best_model['accuracy'],
        'parameters': best_model['params']
    }
    
    # Backup old model
    if os.path.exists('device_classifier_model.pkl'):
        os.rename('device_classifier_model.pkl', 'device_classifier_model_backup.pkl')
        print("📦 Backed up old model as device_classifier_model_backup.pkl")
    
    # Save new optimized model
    with open('device_classifier_model.pkl', 'wb') as f:
        pickle.dump(optimized_model_data, f)
    
    print(f"💾 Saved optimized {best_model_name} model!")
    
    return best_model['accuracy'], best_model_name

if __name__ == "__main__":
    accuracy, model_type = optimize_model()
    print(f"\n🎉 Model optimization complete!")
    print(f"   Final accuracy: {accuracy:.3f}")
    print(f"   Model type: {model_type}")
