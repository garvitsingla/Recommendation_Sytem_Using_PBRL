import numpy as np
import pickle
import torch
from Reward_model_PBRL import RewardModel, encode_trajectory

class OfflineEnv(object):
    
    def __init__(self, users_dict, users_history_lens, movies_id_to_movies, state_size, fix_user_id=None):

        self.users_dict = users_dict
        self.users_history_lens = users_history_lens
        self.items_id_to_name = movies_id_to_movies
        
        self.state_size = state_size
        self.available_users = self._generate_available_users()

        self.fix_user_id = fix_user_id

        self.user = fix_user_id if fix_user_id else np.random.choice(self.available_users)
        self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        self.done = False
        self.recommended_items = set(self.items)
        self.done_count = 3000

        # Load reward model and mapping
        self.reward_model = RewardModel(16)
        self.reward_model.load_state_dict(torch.load("reward_model.pth"))
        self.reward_model.eval()

        with open("movie_id_to_idx.pkl", "rb") as f:
            self.movie_id_to_idx = pickle.load(f)

        from Reward_model_PBRL import embedding_matrix as reward_embedding_matrix
        self.embedding_matrix = reward_embedding_matrix  # must match the trained model
        
    def _generate_available_users(self):
        available_users = []
        for i, length in zip(self.users_dict.keys(), self.users_history_lens):
            if length > self.state_size:
                available_users.append(i)
        return available_users
    
    def reset(self):
        self.user = self.fix_user_id if self.fix_user_id else np.random.choice(self.available_users)
        self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        self.done = False
        self.recommended_items = set(self.items)
        return self.user, self.items, self.done
        
    def step(self, action, top_k=False):

        reward = -0.5
        
        if top_k:
            correctly_recommended = []
            rewards = []
            for act in action:
                if act in self.user_items and act not in self.recommended_items:
                    correctly_recommended.append(act)
                    self.items = self.items[1:] + [act]
                self.recommended_items.add(act)

            # Use preference-based reward for current state
            try:
                trajectory = [(item, self.user_items.get(item, 3)) for item in self.items]
                traj_embedding = encode_trajectory(trajectory, self.movie_id_to_idx, self.embedding_matrix)
                score = self.reward_model(traj_embedding).item()
                rewards = [score] * len(action)  # apply same reward to all
            except:
                rewards = [-0.5] * len(action)

            reward = rewards

        else:
            if action in self.user_items and action not in self.recommended_items:
                self.items = self.items[1:] + [action]
            self.recommended_items.add(action)

            try:
                trajectory = [(item, self.user_items.get(item, 3)) for item in self.items]
                traj_embedding = encode_trajectory(trajectory, self.movie_id_to_idx, self.embedding_matrix)
                reward = self.reward_model(traj_embedding).item()
            except:
                reward = -0.5

        if len(self.recommended_items) > self.done_count or \
           len(self.recommended_items) >= self.users_history_lens[self.user - 1]:
            self.done = True

        return self.items, reward, self.done, self.recommended_items



        #     for act in action:
        #         if act in self.user_items.keys() and act not in self.recommended_items:
        #             correctly_recommended.append(act)
        #             rewards.append((self.user_items[act] - 3)/2)
        #         else:
        #             rewards.append(-0.5)
        #         self.recommended_items.add(act)
        #     if max(rewards) > 0:
        #         self.items = self.items[len(correctly_recommended):] + correctly_recommended
        #     reward = rewards

        # else:
        #     if action in self.user_items.keys() and action not in self.recommended_items:
        #         reward = self.user_items[action] -3  # reward
        #     if reward > 0:
        #         self.items = self.items[1:] + [action]
        #     self.recommended_items.add(action)

        # if len(self.recommended_items) > self.done_count or len(self.recommended_items) >= self.users_history_lens[self.user-1]:
        #     self.done = True
            
        # return self.items, reward, self.done, self.recommended_items



    def get_items_names(self, items_ids):
        items_names = []
        for id in items_ids:
            try:
                items_names.append(self.items_id_to_name[str(id)])
            except:
                items_names.append(list(['Not in list']))
        return items_names
