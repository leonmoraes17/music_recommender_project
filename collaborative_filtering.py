import pandas as pd
import dask.dataframe as dd
from scipy.sparse import csr_matrix, save_npz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from data_cleaning import clean_data

# set paths
#input path
song_data_path = 'D:/spotify_recommender/data/cleaned_data.csv'
song_data_path_musicinfo = 'D:/spotify_recommender/data/Music Info.csv'
user_listening_data_path = 'D:/spotify_recommender/data/User Listening History.csv'

#output path
track_ids_save_path = "D:/spotify_recommender/data/track_ids.npy"
filtered_data_save_path = "D:/spotify_recommender/data/collab_filtered_data1.csv"
interaction_matrix_save_path = "D:/spotify_recommender/data/interaction_matrix.npz"

#function to filter the entire song data  to only those songs that the user history playcount has
def filter_songs_data(songs_data :pd.DataFrame, track_ids = list) -> pd.DataFrame:

    filtered_data = songs_data[songs_data['track_id'].isin(track_ids)]
    filtered_data.sort_values(by = 'track_id', inplace=True)
    filtered_data.reset_index(drop=True, inplace=True)
    filtered_data1 = clean_data(filtered_data)
    filtered_data1.to_csv(filtered_data_save_path)

    print("filtered dataset is created with only unique track_ids as per user listneing dataset")
    print("-------------------------------------------------------------------------------------")


def create_interaction_matrix(user_listening_data: dd.DataFrame) -> csr_matrix :
    df = user_listening_data.copy()
    # convert playcount column to float
    df['playcount'] = df['playcount'].astype(np.float64)

    #convert string column - track id and user id to categorical columns which can then be assign codes(integers) which will occupy less space as a matrix
    df = df.categorize(columns =['track_id', 'user_id'])

    #convert user id and track id to cetegoricial integer values so the sparse matrix will occupy less space
    track_mapping = df['track_id'].cat.codes
    user_mapping = df['user_id'].cat.codes

    #get list of trackids .cat.categories.value will return all the actual names of track ids once(unqiue)
    track_ids = df['track_id'].cat.categories.values

    #save the categories  List of unqie track ids are saved
    np.save(track_ids_save_path, track_ids, allow_pickle =  True)

    print("Track_id categorigal dataset is created in DASK")

    #add index columns to dataframe , basically add the categorically codes to the dask df

    df = df.assign(
        user_idx = user_mapping,
        track_idx = track_mapping  # two new columns will be formed in df with user idx and track ids which will have integers which acts as categories

    )

    #creating the interaction matrix now
    interaction_matrix = df.groupby(['track_idx', 'user_idx'])['playcount'].sum().reset_index()
    #compute the interaction matrix
    interaction_matrix = interaction_matrix.compute()
    print("Interaction matrix created")

    #get indices from interaction matrix to form sparse matrix (basically getting the categorical integer number )

    row_indices = interaction_matrix['track_idx']
    col_indices = interaction_matrix['user_idx']
    values = interaction_matrix['playcount']

    # get the shape of sparse matrix
    n_tracks = row_indices.nunique()
    n_users = col_indices.nunique()
    
    # create the sparse matrix
    sparse_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(n_tracks, n_users))

    #Save spart matrix
    save_npz(interaction_matrix_save_path, sparse_matrix)
    print("Sparse Matrix created")

def collaborative_recommendation(song_name, artist_name, track_ids, songs_data, interaction_matrix, k =5 ):
    #lowercase song name
    # song_name= song_name.lower()

    # #lowercase artist name
    # artist_name = artist_name.lower()

    #fetch the row from songs dataset
    song_row = songs_data.loc[(songs_data['name'] == song_name) & (songs_data['artist'] == artist_name)]

    #trackid of input song
    input_track_id = song_row['track_id'].values.item()  #fetches the track id from the filtered song row

    #get index of the trackid
    ind = np.where(track_ids == input_track_id)[0].item()  #track_ids is a NP file created in the create_interaction_matrix function

    #fetch the input vector
    input_array = interaction_matrix[ind]  #interaciton matrix here is the SParse matrix which will be passed at time of exdcuting the function

    #get similarity scores
    similarity_scores = cosine_similarity(input_array, interaction_matrix)

    #index value of top 5 recommendation
    recomendation_indicies = np.argsort(similarity_scores.ravel())[-k-1:][::-1]  #sort the indicies of the array

    #get top k recommendations
    recommendation_track_ids= track_ids[recomendation_indicies]

    # get top scores
    top_scores = np.sort(similarity_scores.ravel())[-k-1:][::-1]
    
    # get the songs from data and print
    scores_df = pd.DataFrame({"track_id":recommendation_track_ids.tolist(),
                            "score":top_scores})
    top_k_songs = (
                    songs_data
                    .loc[songs_data["track_id"].isin(recommendation_track_ids)]
                    .merge(scores_df,on="track_id")
                    .sort_values(by="score",ascending=False)
                    .drop(columns=["track_id","score"])
                    .reset_index(drop=True)
    )
    

    return top_k_songs
    

#MAIN function
def main():
    #loading the user listening dataset
    df_user_listening = dd.read_csv(user_listening_data_path)

    #getting the unique tracks id to reduce our cleaned dataset to only those tracks
    unique_track_ids = df_user_listening['track_id'].unique().compute()  #compute since its a dask df
    unique_track_ids = unique_track_ids.tolist()

    df_songs = pd.read_csv(song_data_path_musicinfo)
    filter_songs_data(df_songs, unique_track_ids)

    #create interaction matrix
    create_interaction_matrix(df_user_listening)


if __name__ == "__main__":
    main()
