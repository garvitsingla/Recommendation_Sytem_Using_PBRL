import pandas as pd

# Define the file path
input_file = [r'C:\Users\garvi\Downloads\gymnasium\Recommender_system_via_deep_RL\ml-1m\users.dat',
              r'C:\Users\garvi\Downloads\gymnasium\Recommender_system_via_deep_RL\ml-1m\movies.dat',
              r'C:\Users\garvi\Downloads\gymnasium\Recommender_system_via_deep_RL\ml-1m\ratings.dat']
output_file = [r'C:\Users\garvi\Downloads\gymnasium\Recommender_system_via_deep_RL\dataset_csv\users.csv',
               r'C:\Users\garvi\Downloads\gymnasium\Recommender_system_via_deep_RL\dataset_csv\movies.csv',
               r'C:\Users\garvi\Downloads\gymnasium\Recommender_system_via_deep_RL\dataset_csv\ratings.csv']

columns_list = [
    ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
    ['MovieID', 'Title', 'Genres'],
    ['UserID', 'MovieID', 'Rating', 'Timestamp']
]


# Read the .dat file with '::' as separator
for i in range(len(input_file)):
    df = pd.read_csv(input_file[i], sep='::', engine='python', names=columns_list[i], encoding='latin-1')
    df.to_csv(output_file[i], index=False)
# Save to CSV
# df.to_csv(output_file, index=False)

print(f"File successfully converted and saved as {output_file}")
