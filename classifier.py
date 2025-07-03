#%% Package 


import pandas as pd
print(pd.__version__)
import numpy as np
print(np.__version__)
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from collections import defaultdict



#%% prepare data 

##################
### Load files ###
##################

# Load the datasets
df = pd.read_csv("file/path/df.csv")
df2 = pd.read_csv("file/path/df_scaled_classification.csv")

#########################
### Cluster remapping ###
#########################

# Define mapping
cluster_mapping = {
    0: 0,
    3: 1,
    4: 2,
    2: 3,
    6: 4,
    5: 5,
    1: 6
}

# Apply the mapping
df["Assigned_Cluster"] = df2["Assigned_Cluster"].replace(cluster_mapping)

# Reorder columns 
cols = ['Dataset', 'Mouse', 'Person', 'Injury', 'Assigned_Cluster'] + \
       [col for col in df.columns if col not in ['Dataset', 'Mouse', 'Person', 'Injury','Assigned_Cluster']]
df = df[cols]

# Define substrings to search for for cosine/sine transformation
angle_keywords = [
    "mean angle value",
    "mean angle during stance",
    "mean angle during swing",
    "mean propulsion vector angle ",
    "mean phase value",
    "mean phase during stance",
    "mean phase during swing"

]

# Iterate through columns to check if they contain any of the substrings
for col in df.columns:
    if any(keyword in col.lower() for keyword in angle_keywords):
        # Calculate sine and cosine values
        df[col + '_sin'] = np.sin(np.deg2rad(df[col]))  # Convert degrees to radians before sin
        df[col + '_cos'] = np.cos(np.deg2rad(df[col]))  # Convert degrees to radians before cos
        
        # Remove the original column
        df = df.drop(columns=[col])


########################
### Standardize Data ### 
########################
# Store identifiers (not scaled)
identifiers = df[["Dataset","Mouse", "Person", "Injury","Assigned_Cluster"]].reset_index(drop=True)

# Drop unwanted columns and scale
df_numeric = df.drop(columns=["Dataset", "Mouse", "Person", "Injury","Assigned_Cluster",
                              "Number of runs (#)_left_forelimb",
                              "Number of runs (#)_right_forelimb",
                              "Number of runs (#)_left_hindlimb",
                              "Number of runs (#)_right_hindlimb",
                              "Number of steps (#)_right_forelimb",
                              "Number of steps (#)_left_forelimb",
                              "Number of steps (#)_right_hindlimb",
                              "Number of steps (#)_left_hindlimb"],
                    errors="ignore")

df_numeric = df_numeric.select_dtypes(include=["number"])


# Normalize features within each test person
df_numeric_scaled = df.groupby("Person")[df_numeric.columns].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Combine scaled data and identifiers
df_scaled_person = pd.concat([identifiers, df_numeric_scaled], axis=1)


#%% mutual information
##########################
### Mutual information ###
##########################

X = df_scaled_person.drop(columns=["Dataset", "Mouse", "Person", "Injury","Assigned_Cluster"])
num_features_before = X.shape[1]

y = df_scaled_person["Assigned_Cluster"]  

# Calculate Mutual Information (feature and target)
mi = mutual_info_classif(X, y)

# Create a DataFrame to inspect MI
mi_df = pd.DataFrame(mi, index=X.columns, columns=["Mutual Information"])
mi_df = mi_df.sort_values(by="Mutual Information", ascending=False)

# Select top 90 features
top_k = 90
selected_features = mi_df.head(top_k).index


num_features_after = len(selected_features)

dropped_features = num_features_before - num_features_after

# Filter the original dataframe to keep only the selected features
X_selected = X[selected_features]

df_selected = pd.concat([df_scaled_person[["Dataset", "Mouse", "Person", "Injury", "Assigned_Cluster"]], X_selected], axis=1)

###########################
### Plot bar plot of mi ###
###########################
mi_scores = mi  

# Define bin edges
bin_edges = np.arange(0, 1.1, 0.1)
bin_labels = [f'{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}' for i in range(len(bin_edges)-1)]

# calculate how many features fall into each bin
feature_counts, _ = np.histogram(mi_scores, bins=bin_edges)

# Create the barplot
plt.figure(figsize=(8, 6))
sns.barplot(x=bin_labels, y=feature_counts, color='steelblue')

# Add MI values above the bars
for i, count in enumerate(feature_counts):
    plt.text(i, count, str(count), ha='center', va='bottom', fontsize=15, color='black')

# Add vertical grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)


plt.title("Number of Features in MI Value Ranges")
plt.xlabel("Mutual Information Ranges")
plt.ylabel("Number of Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("file/path/MI_scaled_person_classification.png", dpi=300)

plt.show()


df_selected.to_csv("file/path/df_scaled_person_classification.csv", index=False)

#%% split in train and test data 

df_selected = pd.read_csv("file/path/df_scaled_person_classification.csv")

x_array = df_selected.drop(columns=['Dataset', 'Mouse', 'Person', 'Injury','Assigned_Cluster']).values.astype(np.float32)
y_array = df_selected["Assigned_Cluster"].values.squeeze()  


# Use stratify to preserve class distribution
x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(
    x_array, y_array,
    test_size=0.20,
    stratify=y_array,  
    random_state=42
)

# Convert to torch tensors after the split
x_train = torch.tensor(x_train_np, dtype=torch.float32)
x_test = torch.tensor(x_test_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.long)
y_test = torch.tensor(y_test_np, dtype=torch.long)


#%% model definitions MLP, SVM, RF and LogReg 

mlp_model = MLPClassifier(
    hidden_layer_sizes=(256, 128),  
    activation='relu',
    solver='adam',
    alpha=1e-4,                     
    learning_rate_init=0.0001,
    max_iter=1000,
    early_stopping=True, 
    random_state=42
)


# Logistic Regression

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    solver='lbfgs',
    penalty='l2',
    C=10,  
    random_state=42
)

# SVM

svm_model = SVC(
    kernel='poly', 
    degree=2, 
    C=1, 
    gamma='scale', 
    class_weight='balanced', 
    random_state=42)



# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)


#%% Train all models on the full training set and evaluate on the test set + F1 values 


test_results = defaultdict(dict)
f1_per_class = {}

# Define your models again
models = {
    "LogReg": lr_model,
    "LogReg+SMOTE": lr_model,
    "SVM": svm_model,
    "SVM+SMOTE": svm_model,
    "RF": rf_model,
    "RF+SMOTE": rf_model,
    "MLP": mlp_model,
    "MLP+SMOTE": mlp_model
}

# with and without SMOTE
for name, model in models.items():
    print(f"\nTraining {name} on full training set...")

    if "+SMOTE" in name:
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42, k_neighbors=3)),
            ('clf', model)
        ])
        pipeline.fit(x_train_np, y_train_np)
        y_train_pred = pipeline.predict(x_train_np)
        y_test_pred = pipeline.predict(x_test_np)
    else:
        model.fit(x_train_np, y_train_np)
        y_train_pred = model.predict(x_train_np)
        y_test_pred = model.predict(x_test_np)

    # Accuracy and F1
    acc_train = accuracy_score(y_train_np, y_train_pred)
    acc_test = accuracy_score(y_test_np, y_test_pred)
    f1_macro = f1_score(y_test_np, y_test_pred, average='macro', zero_division=0)

    # Save overall performance
    test_results[name]["Train Accuracy"] = acc_train * 100
    test_results[name]["Test Accuracy"] = acc_test * 100
    test_results[name]["Test Macro F1"] = f1_macro * 100

    print(f"{name} → Train Accuracy: {acc_train*100:.2f}%, Test Accuracy: {acc_test*100:.2f}%, F1: {f1_macro*100:.2f}%")

    # Get per-class F1 scores
    report = classification_report(y_test_np, y_test_pred, output_dict=True, zero_division=0)
    class_f1s = {f"F1 Class {label}": report[str(label)]['f1-score'] * 100 for label in np.unique(y_test_np)}
    f1_per_class[name] = class_f1s

# Convert to DataFrame
df_test_results = pd.DataFrame(test_results).T.reset_index()
df_test_results.rename(columns={"index": "Model"}, inplace=True)

# dataframe for barplot
df_melted = pd.melt(
    df_test_results,
    id_vars=["Model"],
    value_vars=["Train Accuracy", "Test Accuracy"],
    var_name="Split",
    value_name="Accuracy"
)

# Plot
plt.figure(figsize=(12, 7))
sns.barplot(
    data=df_melted,
    x="Model", y="Accuracy", hue="Split", palette="pastel", capsize=0.1
)

# Annotate bars
ax = plt.gca()
for bars in ax.containers:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f'{height:.1f}%',
            ha='center', va='bottom', fontsize=12
        )


plt.ylim(50, 120)
plt.ylabel("Accuracy (%)", fontsize=16)
plt.xlabel("Model", fontsize=16)
plt.xticks(rotation=45, fontsize=15)
plt.legend(title=None, fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Save
plt.savefig("file/path/train_test_accuracy_barplot.png", dpi=500)
plt.show()

# F1 Table 
# Combine overall test results and per-class F1 into one table
df_summary = pd.DataFrame(test_results).T
df_f1 = pd.DataFrame(f1_per_class).T
# Merge them
df_full = pd.concat([df_summary, df_f1], axis=1)
df_full.index.name = "Model"
df_full.reset_index(inplace=True)






