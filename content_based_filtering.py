import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder   
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from data_cleaning import data_for_content_filtering
from scipy.sparse import save_npz

# Cleaned Data Path
CLEANED_DATA_PATH = 'D:/spotify_recommender/data/cleaned_data.csv'

#cols to transform

# Cols to transform
frequency_encode_cols = ['year']
ohe_cols = ['artist', 'time_signature', 'key'] # catrgorical columns 
tfidf_col = 'tags' #  just pass the column name as a string
standard_scale_cols = ['duration_ms', 'loudness','tempo'] #numerical columns, scaling 
min_max_scale_cols = ["danceability","energy","speechiness","acousticness","instrumentalness","liveness","valence"] # Normalizing features for distance-based models

def train_transform(data) :  #cleaned data frame after dropping unique columns
    """Trains a ColumnTransformer on the provided data and saves the transformer to a file.
    The ColumnTransformer applies the following transformations:
    - Frequency Encoding using CountEncoder on specified columns.
    - One-Hot Encoding using OneHotEncoder on specified columns.
    - TF-IDF Vectorization using TfidfVectorizer on a specified column.
    - Standard Scaling using StandardScaler on specified columns.
    - Min-Max Scaling using MinMaxScaler on specified columns.
    Parameters:
    data (pd.DataFrame): The input data to be transformed.
    Returns:
    None
    Saves:
    transformer.joblib: The trained ColumnTransformer object.
    """

    transformer = ColumnTransformer(transformers= [
        ('frequence_encode', CountEncoder(normalize=True, return_df=True), frequency_encode_cols),
        ("ohe", OneHotEncoder(handle_unknown="ignore"), ohe_cols),
        ("tfidf", TfidfVectorizer(max_features=85), tfidf_col),
        ("standard_scale", StandardScaler(), standard_scale_cols),
        ("min_max_scale", MinMaxScaler(), min_max_scale_cols)
    ],remainder='passthrough',n_jobs=-1,force_int_remainder_cols=False)

    #fitting the transformer
    transformer.fit(data) 

    #save the transformer
    joblib.dump(transformer,'transformer.joblib')

def transform_data(data): #passing the dataframe through the transformer

        #Load the transformer
    transformer= joblib.load('transformer.joblib')

        # passing the transformer onto the dataframe
    transformed_data = transformer.transform(data) 

    return transformed_data
    
def save_transformed_data(transformed_data, savepath) : #saving the transformed data to NPZ format as its a sparse matric

    save_npz(savepath, transformed_data)


def calculate_similarity_scores(input_vector, data):
     """
    Calculate similarity scores between an input vector and a dataset using cosine similarity.
    Args:
        input_vector (array-like): The input vector for which similarity scores are to be calculated.
        data (array-like): The dataset against which the similarity scores are to be calculated.
    Returns:
        array-like: An array of similarity scores.
    """
     similarity_scores = cosine_similarity(input_vector, data)

     return similarity_scores

def content_recommendation( song_name,artist_name,songs_data, transformed_data, k =5):

    # #convert the song to lower case
    # song_name= song_name.lower()
    # #convert artist name to lower
    # artist_name = artist_name.lower()

    #filter out song from data
    song_row = songs_data.loc[(songs_data['name'] == song_name) & (songs_data['artist'] == artist_name)] 
    #get index of song
    song_index = song_row.index[0]
    input_vector = transformed_data[song_index].reshape(1,-1)
    #calculate similarity
    similarity_score = cosine_similarity(input_vector, transformed_data)
    
    #get top K songs
    top_k_songs_index = np.argsort(similarity_score.ravel())[-k-1:][::-1]
    #top k song names
    top_k_songs_names =  songs_data.iloc[top_k_songs_index] # returns df containing all columns having index values in top_k_songs_index
    top_k_list = top_k_songs_names[['name', 'artist','spotify_preview_url']] # fancy indexing, choosing only certain columns
    return top_k_list


def main(data_path):
    #load the data
    data = pd.read_csv(data_path)

    #clean the data ( remove the columns which are not unique)
    data_content_filtering = data_for_content_filtering(data)

    #train the transformer, fit the transformer on the data
    train_transform(data_content_filtering)

    #transform the data
    transformed_data = transform_data(data_content_filtering)
    #Save the transformed data
    save_transformed_data(transformed_data, 'D:/spotify_recommender/data/transformed_data.npz')

if __name__ == "__main__":
    main (CLEANED_DATA_PATH)