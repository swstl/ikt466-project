import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

def extract_data_from_loader(dataloader):
    """
    Extract all data from a PyTorch DataLoader into numpy arrays.
    
    Args:
        dataloader: PyTorch DataLoader
        
    Returns:
        X: Features as numpy array
        y: Labels as numpy array
    """
    X_list = []
    y_list = []
    
    print("Extracting data from DataLoader...")
    for batch_X, batch_y in tqdm(dataloader):
        # Convert to numpy
        X_list.append(batch_X.cpu().numpy())
        y_list.append(batch_y.cpu().numpy())
    
    # Concatenate all batches
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    return X, y

def flatten_mfcc_features(X):
    """
    Flatten MFCC features for decision tree.
    
    Args:
        X: Array of shape (n_samples, time_steps, n_mfcc) or (n_samples, n_mfcc, time_steps)
        
    Returns:
        X_flat: Flattened features (n_samples, n_features)
    """
    if len(X.shape) == 3:
        # Flatten to (n_samples, time_steps * n_mfcc)
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)
    else:
        X_flat = X
    
    print(f"Flattened shape: {X_flat.shape}")
    return X_flat

def extract_statistics_from_mfcc_batch(X):
    """
    Extract statistical features from MFCC batch.
    Works better than flattening for decision trees.
    
    Args:
        X: Array of shape (n_samples, time_steps, n_mfcc) or (n_samples, n_mfcc, time_steps)
        
    Returns:
        Features with statistics
    """
    n_samples = X.shape[0]
    features_list = []
    
    print("Extracting statistical features...")
    for i in tqdm(range(n_samples)):
        mfcc = X[i]
        
        # Ensure shape is (time, n_mfcc)
        if mfcc.shape[0] < mfcc.shape[1]:
            mfcc = mfcc.T
        
        features = []
        # Statistics across time for each MFCC coefficient
        features.extend(np.mean(mfcc, axis=0))
        features.extend(np.std(mfcc, axis=0))
        features.extend(np.max(mfcc, axis=0))
        features.extend(np.min(mfcc, axis=0))
        
        features_list.append(features)
    
    return np.array(features_list)

def train_decision_tree_from_loaders(train_loader, test_loader, use_statistics=True):
    """
    Train decision tree using PyTorch DataLoaders.
    
    Args:
        train_loader: Training data loader
        test_loader: Test data loader
        use_statistics: If True, extract statistics from MFCC. If False, flatten.
    """
    # Extract data from loaders
    print("="*60)
    print("Extracting training data...")
    print("="*60)
    X_train, y_train = extract_data_from_loader(train_loader)
    
    print(f"Train data shape: {X_train.shape}")
    print(f"Train labels shape: {y_train.shape}")
    
    print("\n" + "="*60)
    print("Extracting test data...")
    print("="*60)
    X_test, y_test = extract_data_from_loader(test_loader)
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Process features
    print("\n" + "="*60)
    print("Processing features...")
    print("="*60)
    
    if use_statistics:
        print("Using statistical features (recommended for decision trees)")
        X_train = extract_statistics_from_mfcc_batch(X_train)
        X_test = extract_statistics_from_mfcc_batch(X_test)
    else:
        print("Using flattened features")
        X_train = flatten_mfcc_features(X_train)
        X_test = flatten_mfcc_features(X_test)
    
    print(f"Final train shape: {X_train.shape}")
    print(f"Final test shape: {X_test.shape}")
    
    # Normalize features
    print("\n" + "="*60)
    print("Normalizing features...")
    print("="*60)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train Decision Tree
    print("\n" + "="*60)
    print("Training Decision Tree...")
    print("="*60)
    
    dt = DecisionTreeClassifier(
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        criterion='gini',
        random_state=42
    )
    
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✓ Decision Tree Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Train Random Forest (usually better)
    print("\n" + "="*60)
    print("Training Random Forest...")
    print("="*60)
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"\n✓ Random Forest Accuracy: {accuracy_rf:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf, zero_division=0))
    
    # Feature importance
    print("\n" + "="*60)
    print("Top 10 Most Important Features (Random Forest):")
    print("="*60)
    feature_importance = sorted(
        enumerate(rf.feature_importances_), 
        key=lambda x: x[1], 
        reverse=True
    )[:10]
    for idx, importance in feature_importance:
        print(f"Feature {idx}: {importance:.4f}")
    
    return dt, rf, scaler
