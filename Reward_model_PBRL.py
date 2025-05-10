import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- CONFIG ---
TRAJECTORY_LENGTH = 5
PREFERENCE_MARGIN = 0.5
NUM_PAIRS = 10000
EMBEDDING_DIM = 16

# --- 1. Load and preprocess ratings data ---
ratings_df = pd.read_csv("ratings.csv")  # Ensure it has UserID, MovieID, Rating, Timestamp
ratings_df.sort_values(by=['UserID', 'Timestamp'], inplace=True)
ratings_df.to_csv("sorted_ratings.csv", index=False)


# --- 2. Segment users into fixed-length trajectories ---
def segment_trajectories(df, traj_len):
    user_groups = df.groupby('UserID')
    all_trajectories = []
    for _, group in user_groups:
        movies = group['MovieID'].values
        ratings = group['Rating'].values
        for i in range(0, len(movies) - traj_len + 1, traj_len):
            traj = [(movies[i + j], ratings[i + j]) for j in range(traj_len)]
            all_trajectories.append(traj)
    return all_trajectories

trajectories = segment_trajectories(ratings_df, TRAJECTORY_LENGTH)

# --- 3. Generate inter-user preference pairs ---
def compute_avg_rating(traj):
    return np.mean([r for (_, r) in traj])

def generate_preference_pairs(trajs, margin, num_pairs):
    preference_data = []
    for _ in range(num_pairs):
        t1, t2 = random.sample(trajs, 2)
        r1, r2 = compute_avg_rating(t1), compute_avg_rating(t2)
        if abs(r1 - r2) > margin:
            label = 1 if r1 > r2 else 0
            preference_data.append((t1, t2, label))
    return preference_data

preference_pairs = generate_preference_pairs(trajectories, PREFERENCE_MARGIN, NUM_PAIRS)

# --- 4. Movie Embeddings ---
movie_ids = ratings_df['MovieID'].unique()
movie_id_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}
embedding_matrix = nn.Embedding(len(movie_ids), EMBEDDING_DIM)

# --- 5. Encode trajectory by averaging movie embeddings ---
def encode_trajectory(traj):
    idxs = [movie_id_to_idx[movie] for movie, _ in traj if movie in movie_id_to_idx]
    idx_tensor = torch.tensor(idxs, dtype=torch.long)
    embeddings = embedding_matrix(idx_tensor)  # (len, emb_dim)
    return embeddings.mean(dim=0)

# --- 6. Dataset class ---
class PreferenceDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        t1, t2, label = self.pairs[idx]
        return encode_trajectory(t1), encode_trajectory(t2), torch.tensor(label, dtype=torch.float32)

dataloader = DataLoader(PreferenceDataset(preference_pairs), batch_size=64, shuffle=True)

# --- 7. Reward Model ---
class RewardModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)

reward_model = RewardModel(EMBEDDING_DIM)
optimizer = optim.Adam(reward_model.parameters(), lr=1e-2)
loss_fn = nn.BCEWithLogitsLoss()

# --- 8. Train the reward function ---
if __name__ == "__main__":

    for epoch in range(100):
        total_loss = 0
        for e1, e2, label in dataloader:
            s1 = reward_model(e1)
            s2 = reward_model(e2)
            logits = s1 - s2
            loss = loss_fn(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")


    # --- Save reward model, embedding matrix, and movie ID mapping ---
    torch.save(reward_model.state_dict(), "reward_model.pth")
    torch.save(embedding_matrix.state_dict(), "embedding_matrix.pth")

    import pickle
    with open("movie_id_to_idx.pkl", "wb") as f:
        pickle.dump(movie_id_to_idx, f)

    print("✅ Reward model and dependencies saved successfully.")
