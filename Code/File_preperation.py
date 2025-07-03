

#%% Packetage

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import plotly.express as px
from pandas import read_csv
from sklearn.model_selection import train_test_split
import openpyxl
from sklearn.feature_selection import mutual_info_classif


#%% Create df file 
##################
### Load files ###
##################

# Load the datasets
df_forelimb = pd.read_csv("file/path/sideview_forelimb_mouse_features_2025.csv")
df_hindlimb = pd.read_csv("file/path/sideview_hindlimb_mouse_features_2025.csv")

################
### forelimb ###
################
# Extract person name and mouse number
df_forelimb[['Person', 'Mouse']] = df_forelimb['Mouse'].str.extract(r'(\w+)_Mouse(\d+)')
df_forelimb['Mouse'] = pd.to_numeric(df_forelimb['Mouse'])

################
### hindlimb ###
################
df_hindlimb[['Person', 'Mouse']] = df_hindlimb['Mouse'].str.extract(r'(\w+)_Mouse(\d+)')
df_hindlimb['Mouse'] = pd.to_numeric(df_hindlimb['Mouse'])


######################################################
### put left and right together for the same mouse ### 
######################################################
### forelimb
# Extract the side (Left/Right) from the Dataset column
df_forelimb['Side'] = df_forelimb['Dataset'].str.extract(r'(Left|Right)$')
df_forelimb['Dataset_Base'] = df_forelimb['Dataset'].str.replace(r'_(Left|Right)$', '', regex=True)

# Split into left and right
left_df = df_forelimb[df_forelimb['Side'] == 'Left'].drop(columns=['Side'])
right_df = df_forelimb[df_forelimb['Side'] == 'Right'].drop(columns=['Side'])

# Merge on both Mouse and Dataset_Base
merged = pd.merge(
    left_df,
    right_df,
    on=['Mouse', 'Dataset_Base'],
    how='outer',
    suffixes=('_left', '_right')
)


# clean
merged['Dataset'] = merged['Dataset_Base']
merged['Person'] = merged['Person_left']
merged = merged.drop(columns=['Dataset_Base', 'Dataset_left', 'Dataset_right', 'Person_left', 'Person_right'])

# Reorder columns
cols = ['Dataset','Person', 'Mouse'] + [col for col in merged.columns if col not in ['Dataset','Person', 'Mouse']]
merged = merged[cols]

df_forelimb_real = merged


### Hindlimb
# Extract the side (Left/Right) from the Dataset column
df_hindlimb['Side'] = df_hindlimb['Dataset'].str.extract(r'(Left|Right)$')
df_hindlimb['Dataset_Base'] = df_hindlimb['Dataset'].str.replace(r'_(Left|Right)$', '', regex=True)

# Split into left and right
left_df = df_hindlimb[df_hindlimb['Side'] == 'Left'].drop(columns=['Side'])
right_df = df_hindlimb[df_hindlimb['Side'] == 'Right'].drop(columns=['Side'])

# Merge on both Mouse and Dataset_Base
merged = pd.merge(
    left_df,
    right_df,
    on=['Mouse', 'Dataset_Base'],
    how='outer',
    suffixes=('_left', '_right')
)


# Clean 
merged['Dataset'] = merged['Dataset_Base']
merged['Person'] = merged['Person_left']
merged = merged.drop(columns=['Dataset_Base', 'Dataset_left', 'Dataset_right', 'Person_left', 'Person_right'])

# Reorder columns
cols = ['Dataset','Person', 'Mouse'] + [col for col in merged.columns if col not in ['Dataset','Person', 'Mouse']]
merged = merged[cols]

df_hindlimb_real = merged

# Define mouse identifying columns
identifying_cols = ["Mouse", "Dataset","Person"]

# Rename all other columns by appending '_forelimb' or '_hindlimb'
df_forelimb_real = df_forelimb_real.rename(columns={col: col + "_forelimb" for col in df_forelimb_real.columns if col not in identifying_cols})
df_hindlimb_real = df_hindlimb_real.rename(columns={col: col + "_hindlimb" for col in df_hindlimb_real.columns if col not in identifying_cols})

# Merge the two datasets
df = pd.merge(df_forelimb_real, df_hindlimb_real, on=["Mouse", "Dataset"], how="outer")
df['Person'] = df['Person_x']
df = df.drop(columns=['Person_x', 'Person_y'])

# Reorder columns
cols = ['Dataset','Person', 'Mouse'] + [col for col in df.columns if col not in ['Dataset','Person', 'Mouse']]
df = df[cols]
df = df.fillna(-1)

#####################
### Define Health ###
#####################
# Define the set of mice that should be labeled as "Severe"
severe_mice1 = {2, 12, 34, 26, 58, 88, 195, 180, 166}
severe_mice2 = {34, 88, 195, 180, 166}

# Create the Injury column
df["Injury"] = np.nan
df["Injury"] = df["Injury"].astype(object)

# Assign Healthy if 'Dataset' contains 'PreSCI'
df.loc[df["Dataset"].str.contains("PreSCI", case=False, na=False), "Injury"] = "Healthy"

# Assign Severe
df.loc[
    (df["Dataset"].str.contains("PostSCI", case=False, na=False)) &
    (df["Mouse"].isin(severe_mice1)),
    "Injury"
] = "Severe"

# Reorder columns
cols = ['Dataset','Person', 'Mouse', 'Injury'] + [col for col in df.columns if col not in ['Dataset','Person', 'Mouse', 'Injury']]
df = df[cols]


df.to_csv("file/path/df.csv", index=False)

#%% normal standerdization (not used in thesis)
########################
### Standardize Data ### 
########################

# first do cosine/sine transformation
# Define substrings to search for
angle_keywords = [
    "mean angle value",
    "mean angle during stance",
    "mean angle during swing",
    "mean propulsion vector angle ",
    "mean phase value",
    "mean phase during stance",
    "mean phase during swing"

]

for col in df.columns:
    if any(keyword in col.lower() for keyword in angle_keywords):
        df[col + '_sin'] = np.sin(np.deg2rad(df[col]))  
        df[col + '_cos'] = np.cos(np.deg2rad(df[col])) 
        # Remove the original column
        df = df.drop(columns=[col])



# Store identifiers (not scaled)
identifiers = df[["Dataset","Mouse", "Person", "Injury"]].reset_index(drop=True)

# Drop unwanted columns and scale
df_numeric = df.drop(columns=["Dataset", "Mouse", "Person", "Injury",
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
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# Combine scaled data and identifiers
df_scaled = pd.DataFrame(df_scaled, columns=df_numeric.columns)
df_scaled = pd.concat([identifiers, df_scaled], axis=1)


##########################
### Mutual information ###
##########################

X = df_scaled.drop(columns=["Dataset", "Mouse", "Person", "Injury"]) 
num_features_before = X.shape[1]


# Map labels
label_mapping = {"Healthy": 0, "Severe": 1}
df_scaled["Injury_Label"] = df_scaled["Injury"].map(label_mapping).fillna(-1).astype(int)

# Reorder columns
cols = ['Dataset', 'Mouse', 'Person', 'Injury', 'Injury_Label'] + \
       [col for col in df_scaled.columns if col not in ['Dataset', 'Mouse', 'Person', 'Injury', 'Injury_Label']]
df_scaled = df_scaled[cols]

y = df_scaled["Injury_Label"]  


# Keep only labeled samples for MI calculation
labeled_mask = df_scaled["Injury_Label"] != -1
X_labeled = X[labeled_mask]
y_labeled = y[labeled_mask]

# Calculate Mutual Information only on labeled data
mi = mutual_info_classif(X_labeled, y_labeled)


# Create a DataFrame to inspect MI values
mi_df = pd.DataFrame(mi, index=X.columns, columns=["Mutual Information"])
mi_df = mi_df.sort_values(by="Mutual Information", ascending=False)

# Select a threshold for MI 
threshold = 0.1  

# Select features with MI above the threshold
selected_features = mi_df[mi_df["Mutual Information"] >= threshold].index

num_features_after = len(selected_features)
dropped_features = num_features_before - num_features_after
print(f"Number of features dropped: {dropped_features}")

# Filter your original dataframe to keep only the selected features
X_selected = X[selected_features]

df_selected = pd.concat([df_scaled[["Dataset", "Mouse", "Person", "Injury", "Injury_Label"]], X_selected], axis=1)

####################
### Plot barplot ###
####################
mi_scores = mi  

bin_edges = np.arange(0, 1.1, 0.1)  
bin_labels = [f'{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}' for i in range(len(bin_edges)-1)]

feature_counts, _ = np.histogram(mi_scores, bins=bin_edges)

# Create the barplot using seaborn
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
plt.savefig("file/path/MI_scaled.png", dpi=300)

plt.show()

df_selected.to_csv("file/path/df_scaled.csv", index=False)

del df_forelimb
del df_hindlimb
del severe_mice1
del severe_mice2


#%% Data standerdized by test performer

##################
### Load files ###
##################

# Load the datasets
df = pd.read_csv("/Users/cilia/OneDrive/Desktop/MasterThesis/data/df.csv")

## Cosine/sine tranformation
# Define substrings to search for
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
        df[col + '_sin'] = np.sin(np.deg2rad(df[col]))  
        df[col + '_cos'] = np.cos(np.deg2rad(df[col]))  
        # Remove the original column
        df = df.drop(columns=[col])


########################
### Standardize Data ### 
########################
# Store identifiers (not scaled)
identifiers = df[["Dataset","Mouse", "Person", "Injury"]].reset_index(drop=True)

# Drop unwanted columns and scale
df_numeric = df.drop(columns=["Dataset", "Mouse", "Person", "Injury",
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

# standerdize features within each test person
df_numeric_scaled = df.groupby("Person")[df_numeric.columns].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Combine scaled data and identifiers
df_scaled_person = pd.concat([identifiers, df_numeric_scaled], axis=1)

##########################
### Mutual information ###
##########################

X = df_scaled_person.drop(columns=["Dataset", "Mouse", "Person", "Injury"])  
num_features_before = X.shape[1]


# Map labels
label_mapping = {"Healthy": 0, "Severe": 1}
df_scaled_person["Injury_Label"] = df_scaled_person["Injury"].map(label_mapping).fillna(-1).astype(int)

# Reorder columns
cols = ['Dataset', 'Mouse', 'Person', 'Injury', 'Injury_Label'] + \
       [col for col in df_scaled_person.columns if col not in ['Dataset', 'Mouse', 'Person', 'Injury', 'Injury_Label']]
df_scaled_person = df_scaled_person[cols]

y = df_scaled_person["Injury_Label"]  
# Keep only labeled samples for MI calculation
labeled_mask = df_scaled_person["Injury_Label"] != -1
X_labeled = X[labeled_mask]
y_labeled = y[labeled_mask]
# Calculate Mutual Information only on labeled data
mi = mutual_info_classif(X_labeled, y_labeled)

# Create a DataFrame to inspect MI values for each feature
mi_df = pd.DataFrame(mi, index=X.columns, columns=["Mutual Information"])
mi_df = mi_df.sort_values(by="Mutual Information", ascending=False)

# Select a threshold for MI 
threshold = 0.1  

# Select features with MI above the threshold
selected_features = mi_df[mi_df["Mutual Information"] >= threshold].index

num_features_after = len(selected_features)
dropped_features = num_features_before - num_features_after
print(f"Number of features dropped: {dropped_features}")

# Filter original dataframe to keep only the selected features
X_selected = X[selected_features]

df_selected = pd.concat([df_scaled_person[["Dataset", "Mouse", "Person", "Injury", "Injury_Label"]], X_selected], axis=1)

###############
### barplot ###
###############

mi_scores = mi

# Define bin edges
bin_edges = np.arange(0, 1.1, 0.1) 
bin_labels = [f'{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}' for i in range(len(bin_edges)-1)]

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
plt.savefig("file/path/MI_scaled_person.png", dpi=300)
plt.show()

df_selected.to_csv("file/path/df_scaled_person.csv", index=False)


#%%


