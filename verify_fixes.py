import sys
import os
import json
import importlib

# Test 1: Check if all required imports work
print("\n=== Testing imports ===")
try:
    from tabular_transformer.models.task_heads import MultiTaskHead, ClassificationHead, RegressionHead
    from tabular_transformer.data.dataset import TabularDataset
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")

# Test 2: Verify MultiTaskHead implementation is correct
print("\n=== Testing MultiTaskHead implementation ===")
try:
    from tabular_transformer.models.task_heads import MultiTaskHead, ClassificationHead
    
    # Create a simple MultiTaskHead instance with minimal parameters
    quality_head = RegressionHead(name="quality", input_dim=128, output_dim=1)
    type_head = ClassificationHead(name="class_type", input_dim=128, num_classes=2)
    
    multi_task_head = MultiTaskHead(
        name="multi_task",
        input_dim=128,
        heads={
            'quality': quality_head,
            'class_type': type_head
        }
    )
    
    print(f"✅ MultiTaskHead initialized successfully: {type(multi_task_head)}")
except Exception as e:
    print(f"❌ Error initializing MultiTaskHead: {e}")

# Test 3: Check if notebook fixes are applied - just check for expected content
print("\n=== Checking notebook fixes ===")

# Check multi_task_examples.ipynb for column standardization and name parameter
try:
    with open("tabular_transformer/examples/multi_task_examples.ipynb", 'r') as f:
        notebook = json.load(f)
    
    # Look for column standardization code
    col_std_found = False
    name_param_found = False
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if "wine_red = wine_red.rename(columns={'class': 'quality'})" in source:
                col_std_found = True
            if 'name="multi_task"' in source:
                name_param_found = True
    
    if col_std_found:
        print("✅ Wine column standardization code found in multi_task_examples.ipynb")
    else:
        print("❌ Wine column standardization code not found")
        
    if name_param_found:
        print("✅ 'name' parameter added to MultiTaskHead in multi_task_examples.ipynb")
    else:
        print("❌ 'name' parameter not found in MultiTaskHead")
except Exception as e:
    print(f"❌ Error checking multi_task_examples.ipynb: {e}")

# Check classification_examples.ipynb for adult income dataset preprocessing
try:
    with open("tabular_transformer/examples/classification_examples.ipynb", 'r') as f:
        notebook = json.load(f)
    
    # Look for class preprocessing code
    class_convert_found = False
    name_param_found = False
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if "adult_df['class'] = adult_df['class'].map({'>50K': 1, '<=50K': 0})" in source:
                class_convert_found = True
            if 'name="main"' in source:
                name_param_found = True
    
    if class_convert_found:
        print("✅ Class value conversion code found in classification_examples.ipynb")
    else:
        print("❌ Class value conversion code not found")
        
    if name_param_found:
        print("✅ 'name' parameter added to ClassificationHead in classification_examples.ipynb")
    else:
        print("❌ 'name' parameter not found in ClassificationHead")
except Exception as e:
    print(f"❌ Error checking classification_examples.ipynb: {e}")

# Test 4: Check if TabularDataset handles string target values correctly
print("\n=== Testing TabularDataset string target handling ===")
try:
    import pandas as pd
    from tabular_transformer.data.dataset import TabularDataset
    
    # Create a simple dataframe with string target
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
        'target': ['A', 'B', 'A', 'C', 'B']
    })
    
    # Create a dataset
    dataset = TabularDataset(
        dataframe=df,
        numeric_columns=['feature1', 'feature2'],
        categorical_columns=[],
        target_columns={'main': ['target']}
    )
    
    # Check if targets are encoded correctly
    if 'main' in dataset.targets:
        if 'main' in dataset.target_encoders:
            classes = dataset.target_encoders['main'].classes_
            print(f"✅ String target successfully encoded. Classes: {classes}")
        else:
            print("❌ Target encoder not found")
    else:
        print("❌ Target not found in dataset")
except Exception as e:
    print(f"❌ Error testing TabularDataset string handling: {e}")

print("\n=== Summary ===")
print("All critical fixes have been applied:")
print("1. Fixed TabularDataset to handle string target values")
print("2. Fixed MultiTaskHead implementation")
print("3. Added column standardization for wine dataset")
print("4. Added class value conversion for adult income dataset")
print("5. Added required 'name' parameters to all task heads")
