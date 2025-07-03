
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
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.functional as F
import plotly.express as px
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder
import umap
from scipy.spatial.distance import cdist
import plotly.express as px
import plotly.io as pio
from collections import defaultdict, Counter
from upsetplot import from_memberships, UpSet
from sklearn.cluster import DBSCAN
from matplotlib.patches import Patch
from scipy.stats import f_oneway, kruskal, shapiro, levene
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp
from itertools import combinations


#%% Data preperation 
#################
### Load data ###
#################

# Load full dataset
df_scaled = pd.read_csv("file/path/df_scaled_person.csv")

########################
### Train test split ###
########################

# Extract features and labels
x_array = df_scaled.drop(columns=['Dataset', 'Mouse', 'Person', 'Injury', 'Injury_Label']).values.astype(np.float32)
y_array = df_scaled["Injury_Label"].values.astype(int)

# Convert to torch tensors
x_tensor = torch.tensor(x_array, dtype=torch.float32)
y_tensor = torch.tensor(y_array, dtype=torch.long)

# Identify labeled and unlabeled data
labeled_mask = y_tensor != -1
x_labeled = x_tensor[labeled_mask]
y_labeled = y_tensor[labeled_mask]
x_unlabeled = x_tensor[~labeled_mask]

# Train-test split on labeled data

# Separate healthy and severe
x_healthy = x_labeled[y_labeled == 0]
y_healthy = y_labeled[y_labeled == 0]
x_severe = x_labeled[y_labeled == 1]
y_severe = y_labeled[y_labeled == 1]

# Split Healthy
x_train_h, x_test_h, y_train_h, y_test_h = train_test_split(
    x_healthy, y_healthy, test_size=0.20, random_state=42
)

# Split Severe
x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(
    x_severe, y_severe, test_size=0.20, random_state=42
)

x_unlabeled_train, x_unlabeled_test = train_test_split(
    x_unlabeled, test_size=0.20, random_state=42
)


# Combine train/test sets
x_train = torch.cat([x_train_h, x_train_s], dim=0)
y_train = torch.cat([y_train_h, y_train_s], dim=0)
x_test = torch.cat([x_test_h, x_test_s], dim=0)
y_test = torch.cat([y_test_h, y_test_s], dim=0)

# Track original indices for tracing later

# Get df_labeled
df_labeled = df_scaled[labeled_mask.numpy()]
df_unlabeled = df_scaled[~labeled_mask.numpy()]

# For Healthy training indices
df_labeled_healthy = df_labeled[df_labeled["Injury_Label"] == 0].reset_index()
train_idx_h, _ = train_test_split(
    df_labeled_healthy.index,
    test_size=0.20,
    random_state=42
)
healthy_indices_in_df_scaled = df_labeled_healthy.loc[train_idx_h, "index"].values

# For Severe training indices
df_labeled_severe = df_labeled[df_labeled["Injury_Label"] == 1].reset_index()
train_idx_s, _ = train_test_split(
    df_labeled_severe.index,
    test_size=0.20,
    random_state=42
)
severe_indices_in_df_scaled = df_labeled_severe.loc[train_idx_s, "index"].values


# Get indices for unlabeled train/test
df_unlabeled = df_unlabeled.reset_index()  
train_idx_u, _ = train_test_split(
    df_unlabeled.index,
    test_size=0.20,
    random_state=42
)
unlabeled_train_indices_in_df_scaled = df_unlabeled.loc[train_idx_u, "index"].values

training_indices_all = np.concatenate([
    healthy_indices_in_df_scaled,
    severe_indices_in_df_scaled,
    unlabeled_train_indices_in_df_scaled   # <-- instead of all unlabeled
])


# Create final dataframe of all training samples
df_training_used = df_scaled.loc[training_indices_all].reset_index(drop=True)

 
#%% create original list for feature grouping 

column_names = df_scaled.columns.tolist()

# Define groups
groups_original = {
    "right forelimb": [],
    "right hindlimb": [],
    "left forelimb": [],
    "left hindlimb": [],
    "tail": [],
    "spine": [],
    "head": [],
    "unknown": []
}

# Assign columns to appropriate groups based on patterns
for col in column_names:
    # === Rule: Right forelimb ===
    if (any(kw in col for kw in ['rforelimb', 'wrist', 'Forepaw', 'elbow', 'shoulder']) and 
        ('_left_forelimb' in col or '_left_hindlimb' in col)):
        groups_original["right forelimb"].append(col)

    # === Rule: Right hindlimb ===
    elif (any(kw in col for kw in ['rhindlimb', 'knee', 'ankle', 'Hindfingers', 'Hindpaw', 'hip']) and 
          ('_left_forelimb' in col or '_left_hindlimb' in col)):
        groups_original["right hindlimb"].append(col)

    # === Rule: Left forelimb ===
    elif (any(kw in col for kw in ['rforelimb', 'wrist', 'Forepaw', 'elbow', 'shoulder']) and 
          ('_right_forelimb' in col or '_right_hindlimb' in col)):
        groups_original["left forelimb"].append(col)

    # === Rule: Left hindlimb ===
    elif (any(kw in col for kw in ['rhindlimb', 'knee', 'ankle', 'Hindfingers', 'Hindpaw', 'hip']) and 
          ('_right_forelimb' in col or '_right_hindlimb' in col)):
        groups_original["left hindlimb"].append(col)

    # === Rule: tail ===
    elif 'tail' in col:
        groups_original["tail"].append(col)

    # === Rule: spine vs base ===
    elif 'spine' in col or 'base' in col:
        spine_index = col.find('spine')
        base_index = col.find('base')

        if spine_index != -1 and base_index != -1:
            if spine_index < base_index:
                groups_original["spine"].append(col)
            else:
                groups_original["tail"].append(col)
        elif 'base' in col:
            groups_original["tail"].append(col)
        elif 'spine' in col:
            groups_original["spine"].append(col)

    # === Rule: head ===
    elif 'head' in col:
        groups_original["head"].append(col)

    # === FINAL RULE: fallback by suffix side ===
    elif '_left_forelimb' in col:
        groups_original["right forelimb"].append(col)
    elif '_right_forelimb' in col:
        groups_original["left forelimb"].append(col)
    elif '_left_hindlimb' in col:
        groups_original["right hindlimb"].append(col)
    elif '_right_hindlimb' in col:
        groups_original["left hindlimb"].append(col)

    # === Unknown
    else:
        groups_original["unknown"].append(col)

# delete unkown becouse there are 0 features ()

if "unknown" in groups_original:
    del groups_original["unknown"]



# Convert group dict to 2-column DataFrame
rows = []
for group, features in groups_original.items():
    feature_str = "; ".join(features)  # Join all features with semicolon
    rows.append({"Group": group, "Features": feature_str})

groups_df_summary = pd.DataFrame(rows)

# Save to CSV
#groups_df_summary.to_csv("file/path/groups_original.csv", index=False)


# Nested dictionary
subgroup_definitions = {
    "Stance phase": ["stance"],
    "Swing Phase": ["swing", "stride", "propulsion", "touchdown"],
    "whole gait": ["Mean", "Number of steps"] 
}


nested_groups_original = {}

for group_name, features in groups_original.items():
    subgroup_dict = {"Stance phase": [], "Swing Phase": [], "whole gait": [], "other": []}
    for feat in features:
        matched = False
        for subgroup, keywords in subgroup_definitions.items():
            if any(kw.lower() in feat.lower() for kw in keywords):
                subgroup_dict[subgroup].append(feat)
                matched = True
                break
        if not matched:
            subgroup_dict["other"].append(feat)
    nested_groups_original[group_name] = subgroup_dict

#%% semisupervised model

########################################
### Semi-Supervised Model Definition ###
########################################

class CVAE(nn.Module):
    def __init__(self, input_dim=7891, label_dim=2, latent_dim=32):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.label_dim = label_dim
       

        # ----- Encoder -----
        self.encoder_fc1 = nn.Linear(input_dim + label_dim, 1024)
        self.encoder_ln1 = nn.LayerNorm(1024)

        self.encoder_fc1_uncond = nn.Linear(input_dim, 1024)
        self.encoder_ln1_uncond = nn.LayerNorm(1024)

        self.encoder_fc2 = nn.Linear(1024, 256)
        self.encoder_ln2 = nn.LayerNorm(256)
        

        self.encoder_mu = nn.Linear(256, latent_dim)
        self.encoder_logvar = nn.Linear(256, latent_dim)

        # ----- Decoder -----
        self.decoder_fc1 = nn.Linear(latent_dim + label_dim, 256)
        self.decoder_ln1 = nn.LayerNorm(256)

        self.decoder_fc1_uncond = nn.Linear(latent_dim, 256)
        self.decoder_ln1_uncond = nn.LayerNorm(256)
    
        self.decoder_fc2 = nn.Linear(256, 1024)
        self.decoder_ln2 = nn.LayerNorm(1024)    

        self.decoder_out = nn.Linear(1024, input_dim)

        # ----- Classifier -----
        self.classifier_fc1 = nn.Linear(latent_dim, 64)
        self.classifier_fc2 = nn.Linear(64, 32)
        self.classifier_out = nn.Linear(32, label_dim)

    def encode(self, x, y=None):
        if y is not None:
            x = torch.cat([x, y], dim=1)
            h = F.relu(self.encoder_ln1(self.encoder_fc1(x)))
        else:
            h = F.relu(self.encoder_ln1_uncond(self.encoder_fc1_uncond(x)))
        
        h = F.relu(self.encoder_ln2(self.encoder_fc2(h)))

        return self.encoder_mu(h), self.encoder_logvar(h)

    def decode(self, z, y=None):
        if y is not None:
            h = F.relu(self.decoder_ln1(self.decoder_fc1(torch.cat([z, y], dim=1))))
        else:
            h = F.relu(self.decoder_ln1_uncond(self.decoder_fc1_uncond(z)))

        h = F.relu(self.decoder_ln2(self.decoder_fc2(h)))

        return self.decoder_out(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def classify(self, z):
        h = F.relu(self.classifier_fc1(z))
        h = F.relu(self.classifier_fc2(h))
        return self.classifier_out(h)

    def forward(self, x, y=None):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, y)
        y_logits = self.classify(z)
        return x_recon, mu, logvar, y_logits

#%%  Training model 

model = CVAE(input_dim=7891, label_dim=2, latent_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 1000000
loss_history = []
kl_history = []
recon_test_history = []
kl_test_history = []
clf_test_history = []
clf_loss_history = []
total_test_loss_history = []
test_accuracy_history = []
recon_loss_history = []
best_loss = float("inf")
patience = 30        
counter = 0
early_stop = False

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
   
    # ----- Supervised (labeled) -----
    class_counts = torch.bincount(y_train, minlength=2)
    class_weights = 1.0 / (class_counts.float() + 1e-6)

    # Map Healthy=0, Severe=1 for classifier
    y_train_onehot = F.one_hot(y_train, num_classes=2).float()
    mu_l, logvar_l = model.encode(x_train, y_train_onehot)
    z_l = model.reparameterize(mu_l, logvar_l)
    x_recon_l = model.decode(z_l, y_train_onehot)
    logits_l = model.classify(mu_l)
    clf_loss = F.cross_entropy(logits_l, y_train, weight=class_weights)

    # ----- Unsupervised (unlabeled) -----
    mu_u, logvar_u = model.encode(x_unlabeled_train, y=None)
    z_u = model.reparameterize(mu_u, logvar_u)
    x_recon_u = model.decode(z_u, y=None) 

    # ----- Combine labeled + unlabeled for reconstruction and KL -----
    mu = torch.cat([mu_l, mu_u], dim=0)
    logvar = torch.cat([logvar_l, logvar_u], dim=0)
    x_recon = torch.cat([x_recon_l, x_recon_u], dim=0)
    x_target = torch.cat([x_train, x_unlabeled_train], dim=0)

# reconstroction loss
    recon_loss = F.mse_loss(x_recon, x_target, reduction='mean')
    
# kl loss with free bits 
    beta_kl=1
    free_bits = 0.5
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl_per_sample = kl_per_dim.sum(dim=1)
    kl_loss = beta_kl * kl_per_sample.mean()  
    
# ----- Total loss -----
    loss = recon_loss + kl_loss + clf_loss
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item()) 
    kl_history.append(kl_loss.item())
    recon_loss_history.append(recon_loss.item())
    clf_loss_history.append(clf_loss.item())

    
    model.eval()
    with torch.no_grad():
        # ----- Labeled test data -----
        y_test_onehot = F.one_hot(y_test, num_classes=2).float()

        mu_test_l, logvar_test_l = model.encode(x_test, y_test_onehot)
        z_test_l = model.reparameterize(mu_test_l, logvar_test_l)
        x_recon_test_l = model.decode(z_test_l, y_test_onehot)
        logits_test_l = model.classify(mu_test_l)
    
        # Losses for labeled
        recon_test_l = F.mse_loss(x_recon_test_l, x_test, reduction='mean')
        kl_test_l = -0.5 * (1 + logvar_test_l - mu_test_l.pow(2) - logvar_test_l.exp())
        kl_test_l = torch.clamp(kl_test_l, min=free_bits).sum(dim=1).mean()
        clf_loss_test = F.cross_entropy(logits_test_l, y_test, weight=class_weights)
        acc_test = (torch.argmax(logits_test_l, dim=1) == y_test).float().mean().item()

        # ----- Unlabeled test data -----
        mu_test_u, logvar_test_u = model.encode(x_unlabeled_test, y=None)
        z_test_u = model.reparameterize(mu_test_u, logvar_test_u)
        x_recon_test_u = model.decode(z_test_u, y=None)
    
        # Losses for unlabeled
        recon_test_u = F.mse_loss(x_recon_test_u, x_unlabeled_test, reduction='mean')
        kl_test_u = -0.5 * (1 + logvar_test_u - mu_test_u.pow(2) - logvar_test_u.exp())
        kl_test_u = torch.clamp(kl_test_u, min=free_bits).sum(dim=1).mean()
    
        # ----- Weighted average over all samples -----
        n_test_l = len(x_test)
        n_test_u = len(x_unlabeled_test)
        n_test_total = n_test_l + n_test_u
    
        recon_test_total = (recon_test_l * n_test_l + recon_test_u * n_test_u) / n_test_total
        kl_test_total = (kl_test_l * n_test_l + kl_test_u * n_test_u) / n_test_total
        total_test_loss = recon_test_total + kl_test_total + clf_loss_test  # classifier only for labeled
    
    # Logging
    recon_test_history.append(recon_test_total.item())
    kl_test_history.append(kl_test_total.item())
    clf_test_history.append(clf_loss_test.item())
    total_test_loss_history.append(total_test_loss.item())
    test_accuracy_history.append(acc_test)
  

    # Early stopping logic
    if total_test_loss.item() < best_loss:
        best_loss = total_test_loss.item()
        counter = 0
        torch.save(model.state_dict(), "file/path/new_cvae_LatentSpace_best_scaled_person.pth")
    else:
        counter += 1
        if counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}. No improvement for {patience} epochs.")
            break
    
    # Logging
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f} | Clf: {clf_loss.item():.4f} | Test Loss: {total_test_loss.item():.4f}")


plt.figure(figsize=(10, 5))
plt.plot(loss_history, label="Train Loss")
plt.plot(total_test_loss_history, '--', label="Test Loss")
plt.title("Total Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("file/path/new_Loss_Train_vs_Test_scaled_person.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(kl_history, label="Train KL")
plt.plot(kl_test_history, '--', label="Test KL")
plt.title("KL Divergence Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("KL Divergence")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("file/path/new_KL_Train_vs_Test_scaled_person.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(recon_loss_history, label="Train recon loss")
plt.plot(recon_test_history, '--', label="Test recon loss")
plt.title("Reconstruction Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Reconstruction Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("file/path/new_recon_Train_vs_Test_scaled_person.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(clf_loss_history, label="Train clf loss")
plt.plot(clf_test_history, '--', label="Test clf loss")
plt.title("Classification Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Classification Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("file/path/new_clf_Train_vs_Test_scaled_person.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(test_accuracy_history)
plt.title("Test Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.tight_layout()
plt.savefig("file/path/new_Test_Accuracy_Over_Epochs_scaled_person.png", dpi=300)
plt.show()


# Save model
torch.save(model.state_dict(), "file/path/new_cvae_LatentSpace_scaled_person.pth")

#%% load model and plot umaps 
# Redefine model
model = CVAE(input_dim=7891, label_dim=2, latent_dim=32)
model.load_state_dict(torch.load("file/path/cvae_LatentSpace_scaled_person.pth"))

model.eval()
mu_all_list = []
label_all_list = []

with torch.no_grad():
    for x, y in zip(torch.cat([x_train, x_unlabeled_train]), torch.cat([y_train, torch.full((x_unlabeled_train.size(0),), -1)])):
        x = x.unsqueeze(0)
        if y.item() != -1:
            y_onehot = F.one_hot(torch.tensor(y), num_classes=2).float().unsqueeze(0)
            mu, _ = model.encode(x, y_onehot)
            label_all_list.append(y.item())
        else:
            mu, _ = model.encode(x, y=None)
            label_all_list.append(-1)
        mu_all_list.append(mu)

mu_all = torch.cat(mu_all_list, dim=0)
laten_mu = mu_all.detach().cpu().numpy()

# Classify
logits_all = model.classify(mu_all)
preds_all = torch.argmax(logits_all, dim=1).numpy()
probs_all = F.softmax(logits_all, dim=1).detach().numpy()

plotly_colors = px.colors.qualitative.Plotly
label_color_map = {
    "Healthy": plotly_colors[0],
    "Severe": plotly_colors[1],
    "Unlabeled": "gray"
}


# 3D UMAP
umap_model1 = umap.UMAP(
    n_components=3,
    n_neighbors=100,
    metric='euclidean',
    min_dist=0.7,
    random_state=42,
    transform_seed=42
)
umap_model1.fit(laten_mu)
umap_proj_all_3d = umap_model1.transform(laten_mu)

# Label names
class_map = {0: "Healthy", 1: "Severe", -1: "Unlabeled"}
label_names = [class_map.get(l, "Other") for l in label_all_list]
predicted_labels = [class_map.get(i, f"Class_{i}") for i in preds_all]

# DataFrame
df_plot1 = pd.DataFrame(umap_proj_all_3d, columns=["UMAP1", "UMAP2", "UMAP3"])
df_plot1["True"] = label_names
df_plot1["Predicted"] = predicted_labels

# True labels view
fig_true = px.scatter_3d(
    df_plot1,
    x="UMAP1", y="UMAP2", z="UMAP3",
    color="True",
    color_discrete_map=label_color_map,
    opacity=1,
    title="3D UMAP of Latent Space (Colored by True Labels)"
)
fig_true.update_traces(marker=dict(size=4, line=dict(width=1, color='black')))
fig_true.update_layout(
    title=dict(
        text="3D UMAP of Latent Space (Colored by True Labels)",
        x=0.5,             # Center title
        xanchor='center',
        yanchor='top',
        font=dict(size=20)
    ),
    legend=dict(
        title="True Label",
        x=0.65,            
        y=0.9,
        xanchor='left',
        bgcolor='rgba(255,255,255,0.7)', 
        bordercolor='black',
        borderwidth=0.5,
        font=dict(size=15)
    ),
    margin=dict(l=10, r=10, b=10, t=50),  
    scene=dict(
        xaxis=dict(title='UMAP1', showgrid=True),
        yaxis=dict(title='UMAP2', showgrid=True),
        zaxis=dict(title='UMAP3', showgrid=True)
    )
    
    
    
)

camera = dict(
    eye=dict(x=4, y=0.5, z=3)
)
fig_true.update_layout(scene_camera=camera)
pio.write_html(fig_true, file="file/path/Latent_UMAP_True_Labels_scaled_person.html", auto_open=True)

######################
### COLOR by Mouse ###
######################

dataset_all = df_training_used["Mouse"].tolist()


umap_model1 = umap.UMAP(
    n_components=3,
    n_neighbors=100,
    metric='euclidean',
    min_dist=0.7,
    random_state=42,
    transform_seed=42
)
umap_model1.fit(laten_mu)
umap_proj_all_3d = umap_model1.transform(laten_mu)

# Label names
class_map = {0: "Healthy", 1: "Severe", -1: "Unlabeled"}
label_names = [class_map.get(l, "Other") for l in label_all_list]

# DataFrame
df_plot1 = pd.DataFrame(umap_proj_all_3d, columns=["UMAP1", "UMAP2", "UMAP3"])
df_plot1["True"] = label_names
df_plot1["Mouse"] = dataset_all

fig_dataset = px.scatter_3d(
    df_plot1,
    x="UMAP1", y="UMAP2", z="UMAP3",
    color="Mouse",
    opacity=1,
    title="3D UMAP of Latent Space (Colored by Dataset)"
)
fig_dataset.update_traces(marker=dict(size=4, line=dict(width=1, color='black')))
fig_dataset.update_layout(margin=dict(l=0, r=0, b=0, t=40))

pio.write_html(fig_dataset, file="file/path/Latent_UMAP_By_Mouse_scaled_person.html", auto_open=True)



############################
### color by test person ###
############################
dataset_all = df_training_used["Person"].tolist()


# 3D UMAP
umap_model1 = umap.UMAP(
    n_components=3,
    n_neighbors=100,
    metric='euclidean',
    min_dist=0.7,
    random_state=42,
    transform_seed=42
)
umap_model1.fit(laten_mu)
umap_proj_all_3d = umap_model1.transform(laten_mu)

# Label names
class_map = {0: "Healthy", 1: "Severe", -1: "Unlabeled"}
label_names = [class_map.get(l, "Other") for l in label_all_list]


# Assign and rename
df_plot1["Person"] = dataset_all
df_plot1["Person"] = df_plot1["Person"].replace({
    "Elisa": "Test Person 1",
    "Brais": "Test Person 2"
})

fig_dataset = px.scatter_3d(
    df_plot1,
    x="UMAP1", y="UMAP2", z="UMAP3",
    color="Person",
    opacity=1,
    color_discrete_map={
        "Test Person 1": "#73026D",  
        "Test Person 2": "#32A401"   
        }
)
fig_dataset.update_traces(marker=dict(size=4, line=dict(width=1, color='black')))
fig_dataset.update_layout(margin=dict(l=0, r=0, b=0, t=40))

fig_dataset.update_layout(
    title=dict(
        text="3D UMAP of Latent Space (Colored by Test Person)",
        x=0.5,             
        xanchor='center',
        yanchor='top',
        font=dict(size=20)
    ),
    legend=dict(
        title="Test Person",
        x=0.65,             
        y=0.9,
        xanchor='left',
        bgcolor='rgba(255,255,255,0.7)', 
        bordercolor='black',
        borderwidth=0.5,
        font=dict(size=15)
    ),
    margin=dict(l=10, r=10, b=10, t=50), 
    scene=dict(
        xaxis=dict(title='UMAP1', showgrid=True),
        yaxis=dict(title='UMAP2', showgrid=True),
        zaxis=dict(title='UMAP3', showgrid=True)
    )
)
pio.write_html(fig_dataset, file="file/path/Latent_UMAP_By_Person_scaled_person.html", auto_open=True)


#%% Assign clusters in laten space 

dbscan = DBSCAN(eps=1.4, min_samples=4)
cluster_all = dbscan.fit_predict(mu_all.detach().cpu().numpy())
df_plot1["Cluster"] = cluster_all
df_plot1["Cluster"] = df_plot1["Cluster"].astype(str)

# Manual color mapping for clusters
cluster_color_map = {
    "0": plotly_colors[0],
    "1": plotly_colors[1],
    "2": plotly_colors[2],
    "3": 'yellow',
    "4": plotly_colors[5],
    "5": plotly_colors[3],
    "-1": 'gray'
}


fig_dataset = px.scatter_3d(
    df_plot1,
    x="UMAP1", y="UMAP2", z="UMAP3",
    color="Cluster",
    color_discrete_map=cluster_color_map,
    opacity=1,
)
fig_dataset.update_traces(marker=dict(size=4, line=dict(width=1, color='black')))
fig_dataset.update_layout(
    title=dict(
        text="3D UMAP of Latent Space (Colored by Clusters)",
        x=0.5,            
        xanchor='center',
        yanchor='top',
        font=dict(size=20)
    ),
    legend=dict(
        title="Cluster Label",
        x=0.60,            
        y=0.9,
        xanchor='left',
        bgcolor='rgba(255,255,255,0.7)', 
        bordercolor='black',
        borderwidth=0.5,
        font=dict(size=15)
    ),
    margin=dict(l=10, r=10, b=10, t=50), 
    scene=dict(
        xaxis=dict(title='UMAP1', showgrid=True),
        yaxis=dict(title='UMAP2', showgrid=True),
        zaxis=dict(title='UMAP3', showgrid=True)
    )
)


camera = dict(
    eye=dict(x=4, y=0.5, z=3)
)
fig_dataset.update_layout(scene_camera=camera)

pio.write_html(fig_dataset, file="file/path/Latent_UMAP_By_Cluster_scaled_person.html", auto_open=True)



# if -1 (outliers) we call it something else (5)
cluster_all_updated = np.copy(cluster_all)  # Create a copy of the original cluster assignments
cluster_all_updated[cluster_all_updated == -1] = 5  # Replace -1 with 5 (assign outliers to cluster 5)

# Get the unique cluster labels 
unique_labels = np.unique(cluster_all_updated)

# Calculate the centroids (mean latent representation) for each cluster
centroids = {}

for label in unique_labels:
    # Get all points in mu_all corresponding to the current cluster
    cluster_points = mu_all[cluster_all_updated == label]
    
    # Calculate the centroid
    centroid = cluster_points.mean(dim=0)  
    centroids[label] = centroid

# Convert centroids to a tensor if necessary
centroids_tensor = torch.stack(list(centroids.values()))

print("Calculated centroids for each cluster:", centroids)


#%% Statistics 

# Add DBSCAN cluster labels
df_training_used["Cluster"] = cluster_all

# Reorder columns
cols = ['Dataset', 'Mouse', 'Person', 'Injury', 'Injury_Label', 'Cluster'] + \
       [col for col in df_training_used.columns if col not in ['Dataset', 'Mouse', 'Person', 'Injury', 'Injury_Label', 'Cluster']]
df_training_used = df_training_used[cols]

# Define feature columns
non_feature_cols = ['Dataset', 'Person', 'Mouse', 'Injury', 'Injury_Label', 'Cluster']
feature_cols = [col for col in df_training_used.columns if col not in non_feature_cols]

# Keep only non-noise clusters (if there is)
df_clustered = df_training_used[df_training_used["Cluster"] != -1].copy()
df_clustered["Cluster"] = df_clustered["Cluster"].astype(int)

# Track assumption violations
violated_features = []

# Run ANOVA or Kruskal with assumption check
p_values = []
feature_names = []

for feature in feature_cols:
    groups = [df_clustered[df_clustered["Cluster"] == c][feature].dropna() for c in sorted(df_clustered["Cluster"].unique())]

    if all(len(g) > 2 for g in groups):  # Minimum for Shapiro
        try:
            # Check normality with Shapiro-Wilk
            normality_pvals = [shapiro(g)[1] for g in groups]
            normality_ok = all(p > 0.05 for p in normality_pvals)

            # Check equal variance with Levene's test
            _, levene_p = levene(*groups)
            levene_ok = levene_p > 0.05

            if not (normality_ok and levene_ok):
                violated_features.append({
                    "Feature": feature,
                    "Normality_OK": normality_ok,
                    "Levene_OK": levene_ok
                })


            
            stat, p = kruskal(*groups)

            p_values.append(p)
            feature_names.append(feature)

        except Exception as e:
            print(f"Error for {feature}: {e}")

# Adjust p-values
rejected, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

df_stats = pd.DataFrame({
    "Feature": feature_names,
    "p_value": p_values,
    "p_adj": pvals_corrected,
    "Significant (FDR < 0.05)": rejected
}).sort_values(by="p_adj")

significant_features = df_stats[df_stats["Significant (FDR < 0.05)"] == True]["Feature"].tolist()

# Posthoc Dunn test on significant features
clusters = sorted(df_clustered["Cluster"].unique())
cluster_pairs = list(combinations(clusters, 2))

posthoc_matrix = pd.DataFrame(index=[f"{i} vs {j}" for i, j in cluster_pairs],
                              columns=significant_features)

for feature in significant_features:
    p_matrix = sp.posthoc_dunn(df_clustered, val_col=feature, group_col='Cluster', p_adjust='fdr_bh')
    for i, j in cluster_pairs:
        try:
            p_val = p_matrix.loc[i, j]
        except KeyError:
            p_val = p_matrix.loc[j, i]
        posthoc_matrix.loc[f"{i} vs {j}", feature] = p_val

posthoc_matrix = posthoc_matrix.astype(float)

# Format for display
posthoc_matrix_formatted = posthoc_matrix.applymap(
    lambda p: f"{p:.4f}*" if p < 0.05 else f"{p:.4f}"
)

# Count significant posthoc features per pair
significant_counts = posthoc_matrix_formatted.apply(lambda row: row.str.contains(r"\*").sum(), axis=1)

# Display summary
for row_label, count in significant_counts.items():
    print(f"{row_label}: {count} significant features")

# Print assumption violation summary
print(f"\n{len(violated_features)} of {len(feature_cols)} features did NOT meet ANOVA assumptions.\n")

# Save everything to Excel
with pd.ExcelWriter("file/path/Cluster_Feature_Analysis.xlsx") as writer:
    df_stats.to_excel(writer, sheet_name="Statistical test on kinematic features", index=False)
    posthoc_matrix.to_excel(writer, sheet_name="Posthoc_Cluster_Pairs")
    pd.DataFrame(violated_features).to_excel(writer, sheet_name="Violated_ANOVA_Assumptions", index=False)




#%% heatmap 1

# Build matrix
count_matrix = pd.DataFrame(0, index=[0,1,2,3,4,5], columns=[0,1,2,3,4,5])

for row_label, count in significant_counts.items():
    i, j = map(int, row_label.split(" vs "))
    count_matrix.loc[i, j] = count
    count_matrix.loc[j, i] = count  

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(count_matrix, annot=True, fmt="d", cmap="Reds", annot_kws={"size": 15})
plt.title("Number of Significantly Different Features Between Cluster Pairs")
plt.xlabel("Cluster",)
plt.ylabel("Cluster")
plt.tight_layout()
plt.savefig("file/path/Number_of_Significantly_Different_Features_Between_Cluster_Pairs.png", dpi=300)
plt.show()


#%% nested loop of significant features 

column_names = posthoc_matrix.columns.tolist()

# Define groups
groups = {
    "right forelimb": [],
    "right hindlimb": [],
    "left forelimb": [],
    "left hindlimb": [],
    "tail": [],
    "spine": [],
    "head": [],
    "unknown": []
}

# Assign columns to appropriate groups based on patterns
for col in column_names:
    # === Rule: Right forelimb ===
    if (any(kw in col for kw in ['rforelimb', 'wrist', 'Forepaw', 'elbow', 'shoulder']) and 
        ('_left_forelimb' in col or '_left_hindlimb' in col)):
        groups["right forelimb"].append(col)

    # === Rule: Right hindlimb ===
    elif (any(kw in col for kw in ['rhindlimb', 'knee', 'ankle', 'Hindfingers', 'Hindpaw', 'hip']) and 
          ('_left_forelimb' in col or '_left_hindlimb' in col)):
        groups["right hindlimb"].append(col)

    # === Rule: Left forelimb ===
    elif (any(kw in col for kw in ['rforelimb', 'wrist', 'Forepaw', 'elbow', 'shoulder']) and 
          ('_right_forelimb' in col or '_right_hindlimb' in col)):
        groups["left forelimb"].append(col)

    # === Rule: Left hindlimb ===
    elif (any(kw in col for kw in ['rhindlimb', 'knee', 'ankle', 'Hindfingers', 'Hindpaw', 'hip']) and 
          ('_right_forelimb' in col or '_right_hindlimb' in col)):
        groups["left hindlimb"].append(col)

    # === Rule: tail ===
    elif 'tail' in col:
        groups["tail"].append(col)

    # === Rule: spine vs base ===
    elif 'spine' in col or 'base' in col:
        spine_index = col.find('spine')
        base_index = col.find('base')

        if spine_index != -1 and base_index != -1:
            if spine_index < base_index:
                groups["spine"].append(col)
            else:
                groups["tail"].append(col)
        elif 'base' in col:
            groups["tail"].append(col)
        elif 'spine' in col:
            groups["spine"].append(col)

    # === Rule: head ===
    elif 'head' in col:
        groups["head"].append(col)

    # === FINAL RULE: fallback by suffix side ===
    elif '_left_forelimb' in col:
        groups["right forelimb"].append(col)
    elif '_right_forelimb' in col:
        groups["left forelimb"].append(col)
    elif '_left_hindlimb' in col:
        groups["right hindlimb"].append(col)
    elif '_right_hindlimb' in col:
        groups["left hindlimb"].append(col)

    # === Unknown
    else:
        groups["unknown"].append(col)

if "unknown" in groups:
    del groups["unknown"]
    
    

# Nested dictionary
subgroup_definitions = {
    "Stance phase": ["stance"],
    "Swing Phase": ["swing", "stride", "propulsion","touchdown"],
    "whole gait": ["Mean", "Number of steps"] 
}


nested_groups = {}

for group_name, features in groups.items():
    subgroup_dict = {"Stance phase": [], "Swing Phase": [], "whole gait": [], "other": []}
    for feat in features:
        matched = False
        for subgroup, keywords in subgroup_definitions.items():
            if any(kw.lower() in feat.lower() for kw in keywords):
                subgroup_dict[subgroup].append(feat)
                matched = True
                break
        if not matched:
            subgroup_dict["other"].append(feat)
    nested_groups[group_name] = subgroup_dict


#%% pie chart


# Count significant features per group for each cluster pair
cluster_pair_rows = posthoc_matrix.index.tolist()
group_counts_per_pair = {}

for row_label in cluster_pair_rows:
    sig_cols = posthoc_matrix.loc[row_label][posthoc_matrix.loc[row_label] < 0.05].index.tolist()
    group_counts = {group: 0 for group in groups.keys()}
    for group, cols in groups.items():
        group_counts[group] = sum(1 for col in sig_cols if col in cols)
    group_counts_per_pair[row_label] = group_counts

# Define consistent colors
group_names = list(groups.keys())
palette = sns.color_palette("muted", len(group_names))
group_colors = {group: palette[i] for i, group in enumerate(group_names)}

# Prepare subplot grid layout (3 rows x 4 columns = 12 total slots)
fig, axs = plt.subplots(3, 4, figsize=(25, 20))
axs = axs.flatten()

# Assign special positions
positions = [(0, 0), (0, 1), (0, 2), (0, 3),  # Row 0
             (1, 0), (1, 1), (1, 2), (1, 3),  # Row 1
             (2, 0), (2, 1), (2, 2), (2, 3)]  # Row 2

# Mapping for pie chart placement 
pie_slots = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11]  
pie_keys = list(group_counts_per_pair.items())

legend_ax = axs[0]
legend_ax.axis('off')
legend_elements = [
    Patch(facecolor=group_colors[group], label=group) for group in group_names
]
legend_ax.legend(
    handles=legend_elements,
    loc="center",
    fontsize=30,
    title="Feature Groups",
    title_fontsize=30,
    frameon=False
)

axs[4].axis('off')  


for idx, ax_idx in enumerate(pie_slots):
    if idx >= len(pie_keys):
        axs[ax_idx].axis('off')
        continue
    pair, group_counts = pie_keys[idx]
    ax = axs[ax_idx]

    # Prepare data
    sizes = []
    colors = []
    for group in group_names:
        count = group_counts[group]
        if count > 0:
            sizes.append(count)
            colors.append(group_colors[group])

    total = sum(sizes)
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        startangle=90,
        autopct=lambda pct: f"{int(round(pct/100.*total))} ({pct:.0f}%)",
        textprops=dict(color="black", fontsize=25)
    )
    ax.set_title(pair, fontsize=30)


plt.suptitle("Significantly Different Features per Group for Each Cluster Pair", fontsize=40)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
# plt.savefig("file/path/PieCharts.png", dpi=300)
plt.show()

#%% Raw significant features heatmap (raw features)
raw_count_matrix = pd.DataFrame(index=group_counts_per_pair.keys(), columns=group_names)

for pair, group_counts in group_counts_per_pair.items():
    for group in group_names:
        raw_count_matrix.loc[pair, group] = group_counts[group]

raw_count_matrix = raw_count_matrix.astype(int)

plt.figure(figsize=(12, 8))
sns.heatmap(
    raw_count_matrix,
    cmap="OrRd",  
    annot=True,
    fmt="d",
    cbar_kws={'label': 'Number of Significantly Different Features'},
    annot_kws={"size": 15}
)
plt.title("Raw Count of Significantly Different Features per Group\n(for Each Cluster Pair)", fontsize=18)
plt.xlabel("Group", fontsize=20)
plt.ylabel("Cluster Pair", fontsize=20)
plt.xticks(rotation=45, ha="right", fontsize=13)
plt.yticks(fontsize=12)
plt.tight_layout()

plt.savefig("file/path/RawCount_Heatmap_Significant_Features.png", dpi=300)
plt.show()


#%% heatmap proportions 
group_total_features = {group: len(cols) for group, cols in groups.items()}

# Compute proportion significant per group per cluster pair
proportion_matrix = pd.DataFrame(index=group_counts_per_pair.keys(), columns=group_names)

for pair, group_counts in group_counts_per_pair.items():
    for group in group_names:
        total_features = group_total_features[group]
        n_sig = group_counts[group]
        proportion = n_sig / total_features if total_features > 0 else 0.0
        proportion_matrix.loc[pair, group] = proportion

proportion_matrix = proportion_matrix.astype(float)

plt.figure(figsize=(12, 8))
sns.heatmap(
    proportion_matrix,
    cmap="Blues",
    annot=True,
    fmt=".2f",
    cbar_kws={'label': 'Proportion of Significantly Different Features'},
    annot_kws={"size": 15}
)
plt.title("Proportion of Significantly Different Features per Group\n(for Each Cluster Pair)", fontsize=18)

plt.xlabel("Group", fontsize=20)
plt.ylabel("Cluster Pair", fontsize=20)
plt.xticks(rotation=45, ha="right", fontsize=13)
plt.yticks(fontsize=12)
plt.tight_layout()

plt.savefig("file/path/Proportion_Heatmap_Significant_Features.png", dpi=300)
plt.show()


#%% nested loop heatmap

nested_group_counts_per_pair = {}

for pair_label in posthoc_matrix.index:
    sig_features = posthoc_matrix.loc[pair_label][posthoc_matrix.loc[pair_label] < 0.05].index.tolist()
    subgroup_counts = {}

    # Loop through nested groups
    for main_group, subgroups in nested_groups.items():
        for subgroup, features in subgroups.items():
            key = f"{main_group} / {subgroup}"
            if key not in subgroup_counts:
                subgroup_counts[key] = 0
            subgroup_counts[key] += sum(1 for f in sig_features if f in features)

    nested_group_counts_per_pair[pair_label] = subgroup_counts

# Build matrix for heatmap
all_nested_keys = sorted({k for d in nested_group_counts_per_pair.values() for k in d})
nested_raw_count_matrix = pd.DataFrame(index=posthoc_matrix.index, columns=all_nested_keys)

for pair, counts in nested_group_counts_per_pair.items():
    for key in all_nested_keys:
        nested_raw_count_matrix.loc[pair, key] = counts.get(key, 0)

nested_raw_count_matrix = nested_raw_count_matrix.astype(int)
nested_raw_count_matrix = nested_raw_count_matrix.drop(columns=[col for col in nested_raw_count_matrix.columns if "other" in col.lower()])

# Plot
plt.figure(figsize=(16, 10))
sns.heatmap(
    nested_raw_count_matrix,
    cmap="OrRd",
    annot=True,
    fmt="d",
    cbar_kws={'label': 'Number of Significantly Different Features'},
    annot_kws={"size": 15}
)
plt.title("Raw Count of Significantly Different Features per Subgroup\n(for Each Cluster Pair)", fontsize=18)
plt.xlabel("Subgroup", fontsize=20)
plt.ylabel("Cluster Pair", fontsize=20)
plt.xticks(rotation=45, ha="right", fontsize=13)
plt.yticks(fontsize=15)
plt.tight_layout()

plt.savefig("file/path/Nested_RawCount_Heatmap_Significant_Features.png", dpi=300)
plt.show()

#%% nested heatmap proportion 

# Count significant features per nested subgroup (using nested_groups_original)
nested_proportion_counts = {}

for pair_label in posthoc_matrix.index:
    sig_features = posthoc_matrix.loc[pair_label][posthoc_matrix.loc[pair_label] < 0.05].index.tolist()
    subgroup_proportions = {}

    for main_group, subgroups in nested_groups_original.items():
        for subgroup, features in subgroups.items():
            key = f"{main_group} / {subgroup}"
            total_features = len(features)
            if total_features > 0:
                n_sig = sum(1 for f in sig_features if f in features)
                proportion = n_sig / total_features
            else:
                proportion = 0.0
            subgroup_proportions[key] = proportion

    nested_proportion_counts[pair_label] = subgroup_proportions

all_nested_keys = sorted({k for d in nested_proportion_counts.values() for k in d})
nested_proportion_matrix = pd.DataFrame(index=posthoc_matrix.index, columns=all_nested_keys)

for pair, proportions in nested_proportion_counts.items():
    for key in all_nested_keys:
        nested_proportion_matrix.loc[pair, key] = proportions.get(key, 0.0)

nested_proportion_matrix = nested_proportion_matrix.astype(float)
nested_proportion_matrix = nested_proportion_matrix.drop(columns=[col for col in nested_proportion_matrix.columns if "other" in col.lower()])

plt.figure(figsize=(18, 10))
sns.heatmap(
    nested_proportion_matrix,
    cmap="Blues",
    annot=True,
    fmt=".2f",
    cbar_kws={'label': 'Proportion of Significantly Different Features'},
    annot_kws={"size": 15}
)
plt.title("Proportion of Significantly Different Features per Subgroup \n(for Each Cluster Pair)", fontsize=18)
plt.xlabel("Subgroup", fontsize=20)
plt.ylabel("Cluster Pair", fontsize=20)
plt.xticks(rotation=45, ha="right", fontsize=13)
plt.yticks(fontsize=15)

plt.tight_layout()
plt.savefig("file/path/Nested_Proportion_Heatmap_Significant_Features.png", dpi=300)
plt.show()



#%% excel summery


# Create cluster_features dict from posthoc_matrix 
# (if not already done)
cluster_ids = sorted(set(int(x) for label in posthoc_matrix.index for x in label.split(" vs ")))
cluster_features = {cid: set() for cid in cluster_ids}

for feature in posthoc_matrix.columns:
    for pair_label in posthoc_matrix.index:
        i, j = map(int, pair_label.split(" vs "))
        if posthoc_matrix.loc[pair_label, feature] < 0.05:
            cluster_features[i].add(feature)
            cluster_features[j].add(feature)

# Map each feature to all clusters it appears in 
feature_membership = defaultdict(set)

for cluster, features in cluster_features.items():
    for f in features:
        feature_membership[f].add(str(cluster))  

# Count each unique cluster combination 
memberships = [tuple(sorted(groups)) for groups in feature_membership.values()]
membership_counter = Counter(memberships)

# create df
df_overlap_summary = pd.DataFrame([
    {
        "Cluster Combination": ", ".join(combo),
        "Clusters": combo,
        "Num Features": count,
    }
    for combo, count in membership_counter.items()
])

df_overlap_summary["Num Clusters"] = df_overlap_summary["Clusters"].apply(len)
df_overlap_summary = df_overlap_summary.sort_values(by=["Num Clusters", "Num Features"], ascending=[False, False])

# Create a new Excel writer object
output_path = "file/path/All_Cluster_Combination_Features.xlsx"
with pd.ExcelWriter(output_path) as writer:
    
    df_overlap_summary.to_excel(writer, sheet_name="Summary", index=False)

    for combo, _ in membership_counter.items():
        combo_str = ", ".join(combo)  # Create a string name for the sheet (e.g., "0,1,2,3")
        feature_list = []  # List to store the features for this combination

        for feature, clusters in feature_membership.items():
            if set(combo) == clusters:
                feature_list.append(feature)

        df_features = pd.DataFrame(feature_list, columns=["Features"])
        df_features.to_excel(writer, sheet_name=combo_str, index=False)



#%% UpSet plot

memberships = [tuple(sorted(combo)) for combo in df_overlap_summary["Clusters"]]
counts = df_overlap_summary["Num Features"].values
series = from_memberships(memberships, data=counts)
series = series.reorder_levels(sorted(series.index.names))  # ensure level order
series = series.sort_index(level=series.index.names[::-1])  # sort each level

# Explicit category order
category_order = ['0', '1', '2', '3', '4', '5']
upset = UpSet(series, sort_by='cardinality', sort_categories_by=None)
upset.intersections.sort_values(ascending=False, inplace=True)
upset._category_order = category_order  # force order
plt.rcParams.update({
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 22
})
fig = plt.figure(figsize=(20, 10))
upset.plot(fig)

# Annotate bars with count values
bar_ax = [ax for ax in fig.axes if ax.get_ylabel() == "Intersection size"]
if bar_ax:
    bar_ax = bar_ax[0]
    for patch in bar_ax.patches:
        height = patch.get_height()
        if height > 0:
            bar_ax.annotate(f"{int(height)}",
                            (patch.get_x() + patch.get_width() / 2, height),
                            ha='center', va='bottom',
                            fontsize=12, color='black')

plt.suptitle("Cluster-Dependent Feature Overlaps", fontsize=16)
plt.tight_layout()
plt.savefig("file/path/Cluster_dependent_feature_overlap.png", dpi=300)

plt.show()


#%% train a classefier in latent space (MLP)
##############################
### TRAIN A NEW CLASSIFIER ###
##############################


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=6):  # output dim = clusters
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#%% Assign pseudolabels from the clusters on the test data 

# test 
# Pass test data through the encoder to get the latent representations
y_test_onehot = F.one_hot(y_test, num_classes=2).float()

mu_test_labeled, _ = model.encode(x_test, y=y_test_onehot)  # For labeled test data, no labels passed
mu_test_unlabeled, _ = model.encode(x_unlabeled_test, y=None)  # For unlabeled test data

# train
# Pass test data through the encoder to get the latent representations
y_train_onehot = F.one_hot(y_train, num_classes=2).float()

mu_train_labeled, _ = model.encode(x_train, y=y_train_onehot)  # For labeled test data, no labels passed
mu_train_unlabeled, _ = model.encode(x_unlabeled_train, y=None)  # For unlabeled test data


# Combine the latent space representations (mu) of training and test data
mu_combined_train_all = torch.cat([mu_train_labeled, mu_train_unlabeled], dim=0)
mu_combined_test_all = torch.cat([mu_test_labeled, mu_test_unlabeled], dim=0)



# Calculate the Euclidean distance between each test data point and the centroids
distances_labeled = torch.cdist(mu_test_labeled, centroids_tensor)  # For labeled test data
distances_unlabeled = torch.cdist(mu_test_unlabeled, centroids_tensor)  # For unlabeled test data

# Assign the closest centroid's cluster as the pseudo-label for each test data point
pseudo_labels_test_labeled = torch.argmin(distances_labeled, dim=1) 
pseudo_labels_test_unlabeled = torch.argmin(distances_unlabeled, dim=1) 

print("Pseudo-labels for labeled test data:", pseudo_labels_test_labeled)
print("Pseudo-labels for unlabeled test data:", pseudo_labels_test_unlabeled)


# TEST

# Combine labeled and pseudo-labeled data for test
x_combined_test = torch.cat([x_test, x_unlabeled_test], dim=0)
y_combined_test = torch.cat([y_test, pseudo_labels_test_unlabeled], dim=0)

# Get the latent representations (mu) using the CVAE encoder
y_combined_test_onehot = F.one_hot(y_combined_test, num_classes=6).float()  



# TRAIN
y_combined_train = torch.tensor(cluster_all_updated, dtype=torch.long)



# Combine labeled and pseudo-labeled data for training
x_combined_train = torch.cat([x_train, x_unlabeled_train], dim=0)

y_combined_train_onehot = F.one_hot(y_combined_train, num_classes=8).float()  





#%% UMAP plot of cluster assignments on traning and test data 

mu_combined_all = torch.cat([mu_combined_train_all, mu_combined_test_all], dim=0)
y_combined_all = torch.cat([y_combined_train, y_combined_test], dim=0)


umap_model = umap.UMAP(
    n_components=3,
    n_neighbors=100,
    metric='euclidean',
    min_dist=0.7,
    random_state=42,
    transform_seed=42
)

# Manual color mapping for clusters
cluster_color_map2 = {
    "Healthy": plotly_colors[0],
    "Severe": plotly_colors[1],
    "Cluster 2": plotly_colors[2],
    "Cluster 3": 'yellow',
    "Cluster 4": plotly_colors[5],
    "Outlier": 'gray'
}


umap_proj = umap_model.fit_transform(mu_combined_all.detach().cpu().numpy())  # Apply UMAP to latent space

df_umap = pd.DataFrame(umap_proj, columns=["UMAP1", "UMAP2", "UMAP3"])
df_umap["Labels"] = y_combined_all.numpy()  

label_map = {0: "Healthy", 1: "Severe", 2:"Cluster 2", 3:"Cluster 3", 4:"Cluster 4", 5: "Outlier"}
df_umap["Label Names"] = df_umap["Labels"].map(label_map)

fig = px.scatter_3d(
    df_umap, 
    x="UMAP1", y="UMAP2", z="UMAP3", 
    color="Label Names", 
    title="3D UMAP of Latent Space (Training + Test Data)",
    color_discrete_map=cluster_color_map2,
    labels={"Label Names": "Cluster Labels"},
  )
    

fig.update_traces(marker=dict(
    size=6, 
    line=dict(width=2, color='black')  
), selector=dict(mode='markers'))

fig.update_traces(marker=dict(
    size=8,  
), selector=dict(mode='markers', marker=dict(color='gray')))


fig.update_layout(
    title=dict(
        text="3D UMAP of Latent Space (Training + Test Data)",
        x=0.5,             
        xanchor='center',
        yanchor='top',
        font=dict(size=20)
    ),
    legend=dict(
        title="Cluster Label",
        x=0.65,            
        y=0.9,
        xanchor='left',
        bgcolor='rgba(255,255,255,0.7)',  
        bordercolor='black',
        borderwidth=0.5,
        font=dict(size=15)
    ),
    margin=dict(l=10, r=10, b=10, t=50),  
    scene=dict(
        xaxis=dict(title='UMAP1', showgrid=True),
        yaxis=dict(title='UMAP2', showgrid=True),
        zaxis=dict(title='UMAP3', showgrid=True)
    )
)


pio.write_html(fig, file="file/path/UMAP_Latent_Space_Train_Test_Visualization.html", auto_open=True)



#%% train MLP classefier on labels and pseudolabels
# Initialize MLP classifier
model_mlp = MLPClassifier(input_dim=32, hidden_dim=128, output_dim=6)  

optimizer = torch.optim.Adam(model_mlp.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Freezing the CVAE model's weights (so it won't be updated during training)
for param in model.parameters():
    param.requires_grad = False

# Training loop
epochs = 10000
total_loss_history_clf = []
total_loss_test_history_clf = []  
accuracy_history_train = []
accuracy_history_test = []

# Early stopping variables
patience = 30  
best_test_loss = float('inf')  
epochs_without_improvement = 0  

for epoch in range(epochs):
    model_mlp.train()  
    optimizer.zero_grad()

    # 1. Pass training data through the encoder to get the latent representations
    y_train_onehot = F.one_hot(y_train, num_classes=2).float()
    mu_train_labeled, _ = model.encode(x_train, y=y_train_onehot)  
    mu_train_unlabeled, _ = model.encode(x_unlabeled_train, y=None) 
    
    # Combine the latent space representations (mu) of training data
    mu_combined_train_all = torch.cat([mu_train_labeled, mu_train_unlabeled], dim=0)
    logits_combined = model_mlp(mu_combined_train_all)  # Pass mu_combined to MLP

    # Use the cluster_all_updated as the pseudo-labels for the unlabeled training data
    y_combined_train = torch.tensor(cluster_all_updated, dtype=torch.long)
    # Use MLP classifier on the latent distribution mean (mu_combined)
    clf_loss = criterion(logits_combined, y_combined_train)  # Compute classification loss
    
    # Total loss 
    total_loss = clf_loss
    total_loss.backward()
    optimizer.step()

    total_loss_history_clf.append(total_loss.item())

    # Calculate training accuracy
    predicted_labels_train = torch.argmax(logits_combined, dim=1)
    accuracy_train = (predicted_labels_train == y_combined_train).float().mean().item()
    accuracy_history_train.append(accuracy_train)

    # Calculate test loss and accuracy
    model_mlp.eval()  # Set model to evaluation mode
    with torch.no_grad():
        
        # 1. Pass test data through the encoder to get the latent representations
        y_test_onehot = F.one_hot(y_test, num_classes=2).float()
        mu_test_labeled, _ = model.encode(x_test, y=y_test_onehot)  # For labeled test data, no labels passed
        mu_test_unlabeled, _ = model.encode(x_unlabeled_test, y=None)  # For unlabeled test data
        
        # Combine the latent space representations (mu) of training and test data
        mu_combined_test_all = torch.cat([mu_test_labeled, mu_test_unlabeled], dim=0)

        y_combined_test = torch.cat([y_test, pseudo_labels_test_unlabeled], dim=0)

        logits_combined_test = model_mlp(mu_combined_test_all)
        clf_loss_test = criterion(logits_combined_test, y_combined_test)  # Compute test classification loss
        total_loss_test_history_clf.append(clf_loss_test.item())

        # Calculate test accuracy
        predicted_labels_test = torch.argmax(logits_combined_test, dim=1)
        accuracy_test = (predicted_labels_test == y_combined_test).float().mean().item()
        accuracy_history_test.append(accuracy_test)

    if epoch % 1 == 0:
        print(f"Epoch {epoch}: Train Loss {total_loss.item():.4f}, Test Loss {clf_loss_test.item():.4f}")
        print(f"Train Accuracy: {accuracy_train * 100:.2f}% | Test Accuracy: {accuracy_test * 100:.2f}%")

    # Early stopping logic
    if clf_loss_test.item() < best_test_loss:
        best_test_loss = clf_loss_test.item()
        epochs_without_improvement = 0  
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered at epoch {epoch}.")
        break

# Plot loss curves for both training and test data
plt.figure(figsize=(10, 5))
plt.plot(total_loss_history_clf, label="Train Loss")
plt.plot(total_loss_test_history_clf, label="Test Loss", linestyle="--")
plt.title("Classification Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.tight_layout()

#plt.savefig("file/path/Loss_Over_Epochs_MLPClassifier.png", dpi=300)
plt.show()


# Plot accuracy curves for both training and test data
plt.figure(figsize=(10, 5))
plt.plot(accuracy_history_train, label="Train Accuracy")
plt.plot(accuracy_history_test, label="Test Accuracy", linestyle="--")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
#plt.savefig("file/path/Accuracy_Over_Epochs_MLPClassifier.png", dpi=300)
plt.show()


# Save model
#torch.save(model_mlp.state_dict(), "file/path/cvae_LatentSpace_MLPClassifier.pth")

#%% proof of concept 

# Redefine CVAE model
model = CVAE(input_dim=7891, label_dim=2, latent_dim=32)
model.load_state_dict(torch.load("file/path/cvae_LatentSpace_scaled_person.pth"))
# Redefine MLP model
model_mlp = MLPClassifier(input_dim=32, hidden_dim=128, output_dim=6)  # 5 clusters
model_mlp.load_state_dict(torch.load("file/path/cvae_LatentSpace_MLPClassifier.pth"))

model.eval()
model_mlp.eval()


# Pass test data through the encoder to get the latent representations
y_test_onehot = F.one_hot(y_test, num_classes=2).float()

mu_test_labeled, _ = model.encode(x_test, y=y_test_onehot)  # For labeled test data, no labels passed
mu_test_unlabeled, _ = model.encode(x_unlabeled_test, y=None)  # For unlabeled test data

# Combine the latent space representations (mu) of training and test data
mu_combined_test_all = torch.cat([mu_test_labeled, mu_test_unlabeled], dim=0)


# Calculate the Euclidean distance between each test data point and the centroids
distances_labeled = torch.cdist(mu_test_labeled, centroids_tensor)  # For labeled test data
distances_unlabeled = torch.cdist(mu_test_unlabeled, centroids_tensor)  # For unlabeled test data

# Assign the closest centroid's cluster as the pseudo-label for each test data point
pseudo_labels_test_labeled = torch.argmin(distances_labeled, dim=1)  # Get the index of the closest centroid
pseudo_labels_test_unlabeled = torch.argmin(distances_unlabeled, dim=1)  # Get the index of the closest centroid

# Combine the latent space representations (mu) of training and test data
test_labels = torch.cat([pseudo_labels_test_labeled, pseudo_labels_test_unlabeled], dim=0)


# `pseudo_labels_labeled` and `pseudo_labels_unlabeled` contain the pseudo-labels for the test data
print("Pseudo-labels for test data:", test_labels)


# TEST

# Combine labeled and pseudo-labeled data for test
x_combined_test = torch.cat([x_test, x_unlabeled_test], dim=0)

mu, _ = model.encode(x_combined_test, y=None)
logits_combined_test = model_mlp(mu_combined_test_all)
pseudo_labels_combined_test = torch.argmax(logits_combined_test, dim=1)


# Compare predicted pseudo-labels with true pseudo-labels (ignoring -1 for unlabeled data)
correct_predictions = (pseudo_labels_combined_test == test_labels)
accuracy = correct_predictions.float().mean().item()

# Output the accuracy
print(f"Accuracy of pseudo-label prediction: {accuracy * 100:.2f}%")



#%%

# Get all features for right forelimb
right_forelimb_features = []
for group, features in nested_groups.get("left hindlimb", {}).items():
    right_forelimb_features.extend(features)

# Initialize summary
feature_summary = []

# Analyze each feature
for feature in right_forelimb_features:
    if feature not in posthoc_matrix.columns:
        continue  # Skip if feature missing
    sig_rows = posthoc_matrix[feature][posthoc_matrix[feature] < 0.05]
    if not sig_rows.empty:
        feature_summary.append({
            "Feature": feature,
            "Significant Comparisons": len(sig_rows),
            "Min P-Value": sig_rows.min(),
            "Cluster Pairs": "; ".join(sig_rows.index)
        })

# Turn into DataFrame
df_summary = pd.DataFrame(feature_summary)
df_summary_hindlimb_left = df_summary.sort_values(by=["Significant Comparisons", "Min P-Value"], ascending=[False, True]).reset_index(drop=True)
