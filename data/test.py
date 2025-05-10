import numpy as np

# Load the .npy file
# user_dict = np.load('user_dict.npy', allow_pickle=True)
user_dict = np.load('users_histroy_len.npy', allow_pickle=True)


# Since it's stored as a 0-dimensional ndarray, you need to extract the actual object
# user_dict = user_dict.item()

# Now it's a real Python dictionary
print(type(user_dict))   # should print: <class 'dict'>
print((user_dict.nunique()))  # now you can access keys properly
