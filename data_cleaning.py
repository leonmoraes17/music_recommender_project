import pandas as pd

DATA_PATH = 'D:/spotify_recommender/data/Music Info.csv'

def clean_data(data):
    # df_songs data will be passed here, the inital dataset with all columns
    """
    Cleans the input DataFrame by performing the following operations:
    1. Removes duplicate rows based on the 'spotify_id' column.
    2. Drops the 'genre' and 'spotify_id' columns.
    3. Fills missing values in the 'tags' column with the string 'no_tags'.
    4. Converts the 'name', 'artist', and 'tags' columns to lowercase.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the data to be cleaned.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    
    return (
        
        data.drop_duplicates(subset = 'spotify_id')
        .drop(columns = ['genre', 'spotify_id']).fillna({'tags': 'no tags'})
        .assign(name = lambda x: x['name'].str.lower(),
                artist = lambda x: x['artist'].str.lower(),
                tags = lambda x: x['tags'].str.lower())
        .reset_index(drop=True)
        
    ) 

print("hello") #the assign function will create new columns name, artist, tags  by passing the dataset through x and converting the specific columns. if dataframe also has the same 
#column names as defined , then the name, artist tags variable columns replaces the one in the original dataset. 


def data_for_content_filtering(data):

    # We are not removing the columns in above  function as we would need that dataframe as well to recomment other songs and hence we would need the name, track id and preview URL
    # The cleaned data from the clean_data function will be passed to this function 
    """
    Cleans the input DataFrame by dropping specific columns.

    This function takes a DataFrame and removes the columns "track_id", "name",
    and "spotify_preview_url". It is intended to prepare the data for content based
    filtering by removing unnecessary features.

    Parameters:
    data (pandas.DataFrame): The input DataFrame containing songs information.

    Returns:
    pandas.DataFrame: A DataFrame with the specified columns removed.
    """
    return (
        data
        .drop(columns=["track_id","name","spotify_preview_url"])
    )

def main(data_path):
    """
    Main function to load, clean, and save data.
    Parameters:
    data_path (str): The file path to the raw data CSV file.
    Returns:
    None
    """
    # load the data
    data = pd.read_csv(data_path)

    #perform data cleaning and return dataframe without genre column and fill tag values 
    cleaned_data = clean_data(data) 

    #save cleaned data
    cleaned_data.to_csv('D:/spotify_recommender/data/cleaned_data.csv', index=False)

if __name__ == "__main__":
        main(DATA_PATH)






