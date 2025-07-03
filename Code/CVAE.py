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




#%% Injury severeity 
#################
### Load data ###
#################
# Load full dataset
df_scaled = pd.read_csv("/Users/cilia/OneDrive/Desktop/MasterThesis/data/df_scaled_person.csv")

# Extract features and labels
x_array = df_scaled.drop(columns=['Dataset', 'Mouse', 'Person', 'Injury', 'Injury_Label']).values.astype(np.float32)
y_array = df_scaled["Injury_Label"].values.astype(int)

# Convert to torch tensors
x_tensor = torch.tensor(x_array, dtype=torch.float32)
y_tensor = torch.tensor(y_array, dtype=torch.long)

labeled_mask = y_tensor != -1


########################
### include covariet ###
########################
df_scaled_with_asym = df_scaled.copy()


# Mice in different clusters
cluster_3_mice = [108, 124, 188, 144, 160, 128, 116, 196, 170, 182, 194, 48, 10, 68, 140, 66, 198, 106, 162, 80, 176, 136, 184]
cluster_4_mice = [114, 190, 20, 112, 120, 192, 168, 146, 152, 130, 172]
cluster_5_mice = [150, 174, 122, 90, 178]
cluster_2_mice = [56, 84, 50, 70, 102, 8, 76, 40, 38, 72, 64, 24, 52, 18, 42, 16, 82, 138, 132, 74, 110, 134, 118, 104, 186, 100, 62, 142, 86, 22, 36, 30, 44, 6, 60, 154, 4, 164, 54, 32, 28, 46, 14, 158, 156, 78]

def assign_cluster(mouse_id, injury_label):
    if injury_label == 0:
        return 0  # Healthy
    elif injury_label == 1:
        return 1  # Severe paralysis
    elif mouse_id in cluster_3_mice:
        return 3
    elif mouse_id in cluster_4_mice:
        return 4
    elif mouse_id in cluster_5_mice:
        return 5
    elif mouse_id in cluster_2_mice:
        return 2
    else:
        return -1  # Unknown or not assigned



df_scaled_with_asym["Cluster"] = df_scaled_with_asym.apply(
    lambda row: assign_cluster(row["Mouse"], row["Injury_Label"]),
    axis=1
)

def assign_asymmetry(cluster, injury_label):
    if injury_label == 0:
        return [0, 0, 0]  # Healthy
    elif cluster == 3:
        return [0, 1, 0]  # Right
    elif cluster == 5:
        return [1, 0, 0]  # Left
    elif cluster in [4, 1]:
        return [0, 0, 1]  # Bilateral
    elif cluster == 2:
        return [0, 0, 1]  # Mild bilateral
    else:
        return [0, 0, 0]  # Unknown

df_scaled_with_asym[["Asym_L", "Asym_R", "Asym_B"]] = df_scaled_with_asym.apply(
    lambda row: pd.Series(assign_asymmetry(row["Cluster"], row["Injury_Label"])),
    axis=1
)




asymmetry_tensor = torch.tensor(df_scaled_with_asym[["Asym_L", "Asym_R", "Asym_B"]].values.astype(np.float32))

# Prepare asymmetry tensors for labeled and unlabeled
asym_labeled = asymmetry_tensor[labeled_mask]
asym_unlabeled = asymmetry_tensor[~labeled_mask]



# Prepare tensors

# Extract features and labels
x_array = df_scaled_with_asym.drop(columns=[
    'Dataset', 'Mouse', 'Person', 'Injury', 'Injury_Label',
    'Cluster', 'Asym_L', 'Asym_R', 'Asym_B'  
]).values.astype(np.float32)
y_array = df_scaled_with_asym["Injury_Label"].values.astype(int)

# Convert to torch tensors
x_tensor = torch.tensor(x_array, dtype=torch.float32)
y_tensor = torch.tensor(y_array, dtype=torch.long)

# Identify labeled and unlabeled data
labeled_mask = y_tensor != -1
x_labeled = x_tensor[labeled_mask]
y_labeled = y_tensor[labeled_mask]
x_unlabeled = x_tensor[~labeled_mask]


########################
### Train-test split ###
########################

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
df_labeled = df_scaled_with_asym[labeled_mask.numpy()]
df_unlabeled = df_scaled_with_asym[~labeled_mask.numpy()]

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
    unlabeled_train_indices_in_df_scaled  
])


# Create final dataframe of all training samples
df_training_used = df_scaled_with_asym.loc[training_indices_all].reset_index(drop=True)







#%% Define CVAE


# Define CVAE2 model
class CVAE2(nn.Module):
    def __init__(self, input_dim=7891, label_dim=2, latent_dim=16):
        super(CVAE2, self).__init__()
        self.encoder_fc1 = nn.Linear(input_dim + label_dim + 3, 1024)
        self.encoder_ln1 = nn.LayerNorm(1024)
        self.encoder_fc1_uncond = nn.Linear(input_dim + 3, 1024)
        self.encoder_ln1_uncond = nn.LayerNorm(1024)
        self.encoder_fc2 = nn.Linear(1024, 256)
        self.encoder_ln2 = nn.LayerNorm(256)
        self.encoder_mu = nn.Linear(256, latent_dim)
        self.encoder_logvar = nn.Linear(256, latent_dim)

        self.decoder_fc1 = nn.Linear(latent_dim + label_dim +3, 256)
        self.decoder_ln1 = nn.LayerNorm(256)
        self.decoder_fc1_uncond = nn.Linear(latent_dim +3, 256)
        self.decoder_ln1_uncond = nn.LayerNorm(256)
        self.decoder_fc2 = nn.Linear(256, 1024)
        self.decoder_ln2 = nn.LayerNorm(1024)
        self.decoder_out = nn.Linear(1024, input_dim)

        self.classifier_fc1 = nn.Linear(latent_dim, 64)
        self.classifier_fc2 = nn.Linear(64, 32)
        self.classifier_out = nn.Linear(32, label_dim)

    def encode(self, x, y=None, asym=None):
        inputs = [x]
        if y is not None:
            inputs.append(y)
        if asym is not None:
            inputs.append(asym)
        x_cat = torch.cat(inputs, dim=1)
    
        if y is not None:
            h = F.relu(self.encoder_ln1(self.encoder_fc1(x_cat)))
        else:
            h = F.relu(self.encoder_ln1_uncond(self.encoder_fc1_uncond(x_cat)))
    
        h = F.relu(self.encoder_ln2(self.encoder_fc2(h)))
        return self.encoder_mu(h), self.encoder_logvar(h)

    def decode(self, z, y=None, asym=None):
        inputs = [z]
        if y is not None:
            inputs.append(y)
        if asym is not None:
            inputs.append(asym)
        h = torch.cat(inputs, dim=1)
    
        if y is not None:
            h = F.relu(self.decoder_ln1(self.decoder_fc1(h)))        
        else:
            h = F.relu(self.decoder_ln1_uncond(self.decoder_fc1_uncond(h))) 
    
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

    def forward(self, x, y=None, asym=None):
        mu, logvar = self.encode(x, y, asym)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, y, asym)
        y_logits = self.classify(z)
        return x_recon, mu, logvar, y_logits


#########################
### Train CVAE2 model ###
#########################

model2 = CVAE2(input_dim=7891)
optimizer = torch.optim.Adam(model2.parameters(), lr=1e-3)

epochs = 100000

loss_history2 = []
kl_history2 = []
recon_test_history2 = []
kl_test_history2 = []
clf_test_history2 = []
clf_loss_history2 = []
total_test_loss_history2 = []
test_accuracy_history2 = []
recon_loss_history2 = []
best_loss = float("inf")
patience = 30       
counter = 0
early_stop = False
free_bits = 0.5

for epoch in range(epochs):
    model2.train()
    optimizer.zero_grad()

    class_counts = torch.bincount(y_labeled, minlength=2)
    class_weights = 1.0 / (class_counts.float() + 1e-6)
    y_labeled_onehot = F.one_hot(y_labeled, num_classes=2).float()

    mu_l, logvar_l = model2.encode(x_labeled, y_labeled_onehot, asym=asym_labeled)
    z_l = model2.reparameterize(mu_l, logvar_l)
    x_recon_l = model2.decode(z_l, y_labeled_onehot, asym=asym_labeled)

    logits_l = model2.classify(mu_l)
    clf_loss = F.cross_entropy(logits_l, y_labeled, weight=class_weights)

    mu_u, logvar_u = model2.encode(x_unlabeled, asym=asym_unlabeled)
    z_u = model2.reparameterize(mu_u, logvar_u)
    x_recon_u = model2.decode(z_u, None, asym=asym_unlabeled)

    mu = torch.cat([mu_l, mu_u], dim=0)
    logvar = torch.cat([logvar_l, logvar_u], dim=0)
    x_recon = torch.cat([x_recon_l, x_recon_u], dim=0)
    x_target = torch.cat([x_labeled, x_unlabeled], dim=0)

    recon_loss = F.mse_loss(x_recon, x_target)
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = torch.clamp(kl_per_dim, min=0.5).sum(dim=1).mean()

    loss = recon_loss + kl_loss + clf_loss
    loss.backward()
    optimizer.step()

    loss_history2.append(loss.item())
    kl_history2.append(kl_loss.item())
    recon_loss_history2.append(recon_loss.item())
    clf_loss_history2.append(clf_loss.item())

    model2.eval()
    with torch.no_grad():
        # Labeled test data 
        y_test_onehot = F.one_hot(y_test, num_classes=2).float()
        asym_test_l = asymmetry_tensor[training_indices_all.shape[0]:][:len(x_test)]
        asym_test_u = asymmetry_tensor[training_indices_all.shape[0]:][len(x_test):]


        mu_test_l, logvar_test_l = model2.encode(x_test, y_test_onehot, asym=asym_test_l)
        z_test_l = model2.reparameterize(mu_test_l, logvar_test_l)
        x_recon_test_l = model2.decode(z_test_l, y_test_onehot, asym=asym_test_l)
        logits_test_l = model2.classify(mu_test_l)
    
        # Losses for labeled
        recon_test_l = F.mse_loss(x_recon_test_l, x_test, reduction='mean')
        kl_test_l = -0.5 * (1 + logvar_test_l - mu_test_l.pow(2) - logvar_test_l.exp())
        kl_test_l = torch.clamp(kl_test_l, min=free_bits).sum(dim=1).mean()
        clf_loss_test = F.cross_entropy(logits_test_l, y_test, weight=class_weights)
        acc_test = (torch.argmax(logits_test_l, dim=1) == y_test).float().mean().item()

        # Unlabeled test data
        mu_test_u, logvar_test_u = model2.encode(x_unlabeled_test, asym=asym_test_u)
        z_test_u = model2.reparameterize(mu_test_u, logvar_test_u)
        x_recon_test_u = model2.decode(z_test_u, None, asym=asym_test_u)
    
        # Losses for unlabeled
        recon_test_u = F.mse_loss(x_recon_test_u, x_unlabeled_test, reduction='mean')
        kl_test_u = -0.5 * (1 + logvar_test_u - mu_test_u.pow(2) - logvar_test_u.exp())
        kl_test_u = torch.clamp(kl_test_u, min=free_bits).sum(dim=1).mean()
    
        # Weighted average over all samples 
        n_test_l = len(x_test)
        n_test_u = len(x_unlabeled_test)
        n_test_total = n_test_l + n_test_u
    
        recon_test_total = (recon_test_l * n_test_l + recon_test_u * n_test_u) / n_test_total
        kl_test_total = (kl_test_l * n_test_l + kl_test_u * n_test_u) / n_test_total
        total_test_loss = recon_test_total + kl_test_total + clf_loss_test  # classifier only for labeled
    
    # Logging
    recon_test_history2.append(recon_test_total.item())
    kl_test_history2.append(kl_test_total.item())
    clf_test_history2.append(clf_loss_test.item())
    total_test_loss_history2.append(total_test_loss.item())
    test_accuracy_history2.append(acc_test)

    if loss.item() < best_loss:
        best_loss = loss.item()
        counter = 0
        torch.save(model2.state_dict(), "cvae2_best.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping.")
            break

    
    # Logging
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f} | Clf: {clf_loss.item():.4f} | Test Loss: {total_test_loss.item():.4f}")


# Plot loss

plt.figure(figsize=(10, 5))
plt.plot(loss_history2, label="Train Loss")
plt.plot(total_test_loss_history2, '--', label="Test Loss")
plt.title("Total Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("file/path/2Loss_Train_vs_Test_scaled_person.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(kl_history2, label="Train KL")
plt.plot(kl_test_history2, '--', label="Test KL")
plt.title("KL Divergence Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("KL Divergence")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("file/path/2KL_Train_vs_Test_scaled_person.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(recon_loss_history2, label="Train recon loss")
plt.plot(recon_test_history2, '--', label="Test recon loss")
plt.title("Reconstruction Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Reconstruction Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("file/path/2recon_Train_vs_Test_scaled_person.png", dpi=300)
plt.show()



plt.figure(figsize=(10, 5))
plt.plot(clf_loss_history2, label="Train clf loss")
plt.plot(clf_test_history2, '--', label="Test clf loss")
plt.title("Classification Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Classification Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("file/path/2clf_Train_vs_Test_scaled_person.png", dpi=300)
plt.show()



plt.figure(figsize=(10, 5))
plt.plot(test_accuracy_history2)
plt.title("Test Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.tight_layout()
plt.savefig("file/path/2Test_Accuracy_Over_Epochs_scaled_person.png", dpi=300)
plt.show()



# Save model
torch.save(model2.state_dict(), "file/path/cvae2_LatentSpace_scaled_person.pth")




#%% UMAP plots 

# Redefine model
model2 = CVAE2(input_dim=7891, label_dim=2, latent_dim=16)
model2.load_state_dict(torch.load("file/path/cvae2_LatentSpace_scaled_person.pth"))


model2.eval()
mu_all_list = []
label_all_list = []


# Stack everything
x_all = torch.cat([x_train, x_unlabeled_train])
y_all = torch.cat([y_train, torch.full((x_unlabeled_train.size(0),), -1)])
asym_all = torch.cat([asym_labeled, asym_unlabeled])

with torch.no_grad():
    for i in range(x_all.size(0)):
        x = x_all[i].unsqueeze(0)
        y = y_all[i].item()
        asym = asym_all[i].unsqueeze(0)

        if y != -1:
            y_onehot = F.one_hot(torch.tensor(y), num_classes=2).float().unsqueeze(0)
            mu, _ = model2.encode(x, y_onehot, asym=asym)
            label_all_list.append(y)
        else:
            mu, _ = model2.encode(x, y=None, asym=asym)
            label_all_list.append(-1)
        mu_all_list.append(mu)
        
    
mu_all = torch.cat(mu_all_list, dim=0)
laten_mu = mu_all.detach().cpu().numpy()

# Classify
logits_all = model2.classify(mu_all)
preds_all = torch.argmax(logits_all, dim=1).numpy()
probs_all = F.softmax(logits_all, dim=1).detach().numpy()

plotly_colors = px.colors.qualitative.Plotly
label_color_map = {
    "Healthy": plotly_colors[0],
    "Severe": plotly_colors[1],
    "Unlabeled": "gray"
}
###########################
### color by true label ###
###########################
umap_model1 = umap.UMAP(
    n_components=3,
    n_neighbors=50,
    metric='euclidean',
    min_dist=0.9,
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
        x=0.65,             # Move legend closer to plot
        y=0.9,
        xanchor='left',
        bgcolor='rgba(255,255,255,0.7)',  # semi-transparent background
        bordercolor='black',
        borderwidth=0.5,
        font=dict(size=15)
    ),
    margin=dict(l=10, r=10, b=10, t=50),  # reduce white space
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
pio.write_html(fig_true, file="file/path/2Latent_UMAP_True_Labels_scaled_person.html", auto_open=True)


######################
### color by mouse ###
######################

# Reuse the training set metadata
dataset_all = df_training_used["Mouse"].tolist()


umap_model1 = umap.UMAP(
    n_components=3,
    n_neighbors=50,
    metric='euclidean',
    min_dist=0.9,
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

pio.write_html(fig_dataset, file="file/path/2Latent_UMAP_By_Mouse_scaled_person.html", auto_open=True)

#%% cluster assignment 

dbscan = DBSCAN(eps=0.745, min_samples=4)  
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
    "6": plotly_colors[4],
    "7": plotly_colors[8],
    "-1": 'gray'
}
#######################################
### Plot umap by cluster assignment ###
#######################################
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
        x=0.90,            
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

pio.write_html(fig_dataset, file="file/path/2Latent_UMAP_By_Cluster_scaled_person.html", auto_open=True)


#####################################################
### Claculate centroid of the cluster assignments ###
#####################################################
cluster_all_updated = np.copy(cluster_all)  # Create a copy of the original cluster assignments
cluster_all_updated[cluster_all_updated == -1] = 7  # Replace -1 with 7 if there are any 

# Get the unique cluster labels 
unique_labels = np.unique(cluster_all_updated)

# Calculate the centroids (mean latent representation) for each cluster
centroids = {}

for label in unique_labels:
    # Get all points in mu_all corresponding to the current cluster
    cluster_points = mu_all[cluster_all_updated == label]
    
    # Calculate the centroid (mean of the cluster points in latent space)
    centroid = cluster_points.mean(dim=0)  # Taking the mean across the samples in the cluster
    centroids[label] = centroid

# Convert centroids to a tensor if necessary
centroids_tensor = torch.stack(list(centroids.values()))



#%% statistics 

# Add DBSCAN cluster labels
df_training_used["Cluster"] = cluster_all

# Reorder columns
cols = ['Dataset', 'Mouse', 'Person', 'Injury', 'Injury_Label', 'Cluster'] + \
       [col for col in df_training_used.columns if col not in ['Dataset', 'Mouse', 'Person', 'Injury', 'Injury_Label', 'Cluster']]
df_training_used = df_training_used[cols]

# Define feature columns
non_feature_cols = ['Dataset', 'Person', 'Mouse', 'Injury', 'Injury_Label', 'Cluster']
feature_cols = [col for col in df_training_used.columns if col not in non_feature_cols]

# Keep only non-noise clusters (if there are any)
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
with pd.ExcelWriter("file/path/Cluster_Feature_Analysis2.xlsx") as writer:
    df_stats.to_excel(writer, sheet_name="Statistical test on kinematic features", index=False)
    posthoc_matrix.to_excel(writer, sheet_name="Posthoc_Cluster_Pairs")
    pd.DataFrame(violated_features).to_excel(writer, sheet_name="Violated_ANOVA_Assumptions", index=False)



#%%

# Build matrix
count_matrix = pd.DataFrame(0, index=[0,1,2,3,4,5,6], columns=[0,1,2,3,4,5,6]) # change if clusterassignments change 

for row_label, count in significant_counts.items():
    i, j = map(int, row_label.split(" vs "))
    count_matrix.loc[i, j] = count
    count_matrix.loc[j, i] = count  

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(count_matrix, annot=True, fmt="d", cmap="Reds", annot_kws={"size": 15})
plt.xlabel("Cluster",)
plt.ylabel("Cluster")
plt.tight_layout()
plt.savefig("file/path/Number_of_Significantly_Different_Features_Between_Cluster_Pairs2.png", dpi=300)
plt.show()



#%% nested loop


# Assuming posthoc_matrix is already loaded
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

# New semantic definitions
subgroup_definitions = {
    "Acceleration": ["acceleration"],
    "Velocity": ["velocity"],
    "Jerk": ["jerk"],
    "Angle": ["angle"],
    "Phase": ["phase"],
    "Length and height": ["length", "step length", "step height"],
    "Duration":["duration"],
    "Frequency": ["frequency"],
}



nested_groups = {}

for group_name, features in groups.items():
    subgroup_dict = {"Acceleration": [], "Velocity": [], "Jerk": [], "Angle":[], "Phase": [], "Length and height":[], "Frequency":[], "Duration":[], "other": []}
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

cluster_pair_rows = posthoc_matrix.index.tolist()
group_counts_per_pair = {}

for row_label in cluster_pair_rows:
    sig_cols = posthoc_matrix.loc[row_label][posthoc_matrix.loc[row_label] < 0.05].index.tolist()
    group_counts = {group: 0 for group in groups.keys()}
    for group, cols in groups.items():
        group_counts[group] = sum(1 for col in sig_cols if col in cols)
    group_counts_per_pair[row_label] = group_counts

group_names = list(groups.keys())
palette = sns.color_palette("muted", len(group_names))
group_colors = {group: palette[i] for i, group in enumerate(group_names)}

fig, axs = plt.subplots(3, 4, figsize=(25, 20))
axs = axs.flatten()

# Assign special positions 
positions = [(0, 0), (0, 1), (0, 2), (0, 3),  # Row 0
             (1, 0), (1, 1), (1, 2), (1, 3),  # Row 1
             (2, 0), (2, 1), (2, 2), (2, 3)]  # Row 2

# Mapping for pie chart placement (skipping legend and one empty spot)
pie_slots = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11]  # Subplot indices where pies go
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
# plt.savefig("file/path/2PieCharts.png", dpi=300)
plt.show()

#%% Raw significant features heatmap
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

plt.savefig("file/path/RawCount_Heatmap_Significant_Features2.png", dpi=300)
plt.show()


#%% proportion of significant feature groups heatmap
group_total_features = {group: len(cols) for group, cols in groups.items()}
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

plt.savefig("file/path/Proportion_Heatmap_Significant_Features2.png", dpi=300)
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

all_nested_keys = sorted({k for d in nested_group_counts_per_pair.values() for k in d})
nested_raw_count_matrix = pd.DataFrame(index=posthoc_matrix.index, columns=all_nested_keys)

for pair, counts in nested_group_counts_per_pair.items():
    for key in all_nested_keys:
        nested_raw_count_matrix.loc[pair, key] = counts.get(key, 0)

nested_raw_count_matrix = nested_raw_count_matrix.astype(int)
nested_raw_count_matrix = nested_raw_count_matrix.drop(columns=[col for col in nested_raw_count_matrix.columns if "other" in col.lower()])

plt.figure(figsize=(16, 10))
sns.heatmap(
    nested_raw_count_matrix,
    cmap="OrRd",
    annot=True,
    fmt="d",
    cbar_kws={'label': 'Number of Significantly Different Features'},
    annot_kws={"size": 15}
)
plt.xlabel("Subgroup", fontsize=20)
plt.ylabel("Cluster Pair", fontsize=20)
plt.xticks(rotation=45, ha="right", fontsize=13)
plt.yticks(fontsize=15)
plt.tight_layout()

plt.savefig("file/path/2Nested_RawCount_Heatmap_Significant_Features.png", dpi=300)
plt.show()




#%% assign pseudolabels 

# Encode test data to get latent representations (mu) 

# One-hot for labeled test
y_test_onehot = F.one_hot(y_test, num_classes=2).float()

# Encode
mu_test_labeled, _ = model2.encode(x_test, y=y_test_onehot, asym=asym_test_l)
mu_test_unlabeled, _ = model2.encode(x_unlabeled_test, y=None, asym=asym_test_u)

# Assign clusters based on nearest centroid 

# Convert to numpy for distance computation
mu_test_labeled_np = mu_test_labeled.detach().cpu().numpy()
mu_test_unlabeled_np = mu_test_unlabeled.detach().cpu().numpy()
centroids_np = centroids_tensor.detach().cpu().numpy()

# Compute distances and assign cluster
assigned_clusters_labeled = np.argmin(cdist(mu_test_labeled_np, centroids_np), axis=1)
assigned_clusters_unlabeled = np.argmin(cdist(mu_test_unlabeled_np, centroids_np), axis=1)


# Recover original DataFrame indices of test samples
# You already had this when splitting train/test

test_idx_h = df_labeled_healthy.index.difference(train_idx_h)
test_idx_s = df_labeled_severe.index.difference(train_idx_s)

test_indices_h = df_labeled_healthy.loc[test_idx_h, "index"].values
test_indices_s = df_labeled_severe.loc[test_idx_s, "index"].values
test_indices_labeled = np.concatenate([test_indices_h, test_indices_s])

# Unlabeled test indices (from earlier split)
test_idx_u = df_unlabeled.index.difference(train_idx_u)
test_indices_unlabeled = df_unlabeled.loc[test_idx_u, "index"].values


##################################################
### Assign clusters to training + test samples ###
##################################################

# Assign training clusters from DBSCAN (cluster_all_updated)
df_scaled_with_asym.loc[training_indices_all, "Assigned_Cluster"] = cluster_all_updated
df_scaled_with_asym.loc[test_indices_labeled, "Assigned_Cluster"] = assigned_clusters_labeled
df_scaled_with_asym.loc[test_indices_unlabeled, "Assigned_Cluster"] = assigned_clusters_unlabeled

df_scaled_with_asym["Assigned_Cluster"] = df_scaled_with_asym["Assigned_Cluster"].astype(int)
df_scaled_classification = df_scaled_with_asym.drop(columns=["Cluster", "Asym_L", "Asym_R", "Asym_B"])

df_scaled_classification.to_csv(
    "file/path/df_scaled_classification.csv",
    index=False
)







#%%
