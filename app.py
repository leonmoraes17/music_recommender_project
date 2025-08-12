import streamlit as st
from content_based_filtering import content_recommendation
from collaborative_filtering import collaborative_recommendation
from scipy.sparse import load_npz
import pandas as pd
import numpy as np
from numpy import load
import pickle
from Hybrid_recommendations import HybridRecommenderSystem as hrs


#Loading all datasets when the page loads.

#transformed data path ; to compare input vector for content based
transformed_data_path = "transformed_data.npz"
transformed_data = load_npz(transformed_data_path)

#cleaned data needed to select song and get the input vector
cleaned_data_path = "cleaned_data.csv"

songs_data = pd.read_csv(cleaned_data_path)



# load track_ids data   
track_ids_path = "track_ids.npy"
track_ids = load(track_ids_path, allow_pickle = True)


#load filtered songs
filtered_data_path = "collab_filtered_data1.csv"
filtered_data = pd.read_csv(filtered_data_path)  #30,000 song dataset

#load interaction sparse matrix
interaction_matrix_path = "interaction_matrix.npz"
interaction_matrix = load_npz(interaction_matrix_path)

#load transformed Hybrid data ( the data 30k which is transformed)
transformed_hybrid_data_path = "transformed_hybrid_data.npz"
transform_hybrid_data = load_npz(transformed_hybrid_data_path)


song_list_df = pickle.load(open('song_db.pkl','rb'))
song_list_names = np.unique(song_list_df['name'].values)
song_list_artist = np.unique(song_list_df['artist'].values)

st.title('Welcome to Music Recommender !')

# Subheader
st.write('### Enter the name of a song and the recommender will suggest similar songs 🎵🎧')

# Text input

song_name = st.selectbox('Type a song name', song_list_names)
st.write('You entered', song_name)

artist_name = st.selectbox("Type artist name - Press enter to continue", song_list_artist)

st.write('You entered:' , artist_name)

# converting to lower case
song_name = song_name.lower()
artist_name= artist_name.lower()

# st.write("Choose type of filtering")
# st.write(" A) Content based filtering will give you recommendations based on similar attributes of the song you searched for")
# st.write(" B) Collaborative filtering will give you songs based on other user preferences i.e what other songs user listens to if they listen to the song you entered ")
# filtering_type = st.selectbox('Select type of filtering:', ['Content-based filtering', 'Collaborative-based filtering', 'Hybrid Recommender System'], index = 2 )

if ((filtered_data['name']==song_name) & (filtered_data['artist']== artist_name)).any():
    #type of filtering
    filtering_type = 'Hybrid Recommender System'  # setting the variable value of 'filtering type tp Hybrid or Content based
      # diversity slider
    diversity = st.slider(label="Diversity in Recommendations",
                        min_value=1,
                        max_value=9,
                        value=5,
                        step=1)

    content_based_weight = 1 - (diversity / 10)

else : 
    filtering_type = 'Content Based System'

#button

if filtering_type == 'Content Based System' :
    if artist_name:
        if st.button('Get Content Recommendation'):  # if the Recommend button is click and filtering type variable value is Content based then do the following below :
            
            if ((songs_data['name'] == song_name) & (songs_data['artist']== artist_name)).any():
                st.write("Content Based Filtering only")
                st.write ('Recommendations for ', f"**{song_name}**")
                
                recommendations = content_recommendation(song_name=song_name,
                                                        artist_name=artist_name,
                                                        songs_data=songs_data,
                                                        transformed_data=transformed_data)  # data is cleaned_data, trnasformed_data is loaded npz data
                # the top 10 songs (name, artist and spotify_url are "Returned" from the function and stored in recommendation variable)

                # display Recommendation 
                for ind, row in recommendations.iterrows():  # iterate through every row of the df which content_recommendation function is returning
                    song_name = row['name'].title()
                    artist_name = row['artist'].title()

                    st.markdown(f'###{ind}.  **{song_name}** by **{artist_name}')
                    st.audio(row['spotify_preview_url'])
                    st.write('----')

            else:
                st.write (f'Sorry , we coudnt find {song_name} in  our database of songs')

elif filtering_type == 'Hybrid Recommender System':
    if artist_name: 
        st.write("Hybrid Based Filering - lower diversity bar to see recommended songs with similar attributes and increase diveristy bar to see which other songs users have played who have listened to -  ", f" {song_name}")
        if st.button('Get Hybrid Recommendations'):
            
            st.write ('Recommendations for ', f"**{song_name}**")
            recommender = hrs(number_of_recommendations=5,
                                weight_content_based=content_based_weight,
                                )
            recommendations = recommender.give_recommendations(song_name=song_name, artist_name=artist_name,
                                                                song_data= filtered_data,                                                               transformed_matrix=transform_hybrid_data,
                                                                track_ids=track_ids,
                                                                interaction_matrix=interaction_matrix)

                # display Recommendation 
            for ind, row in recommendations.iterrows():  # iterate through every row of the df which content_recommendation function is returning
                song_name = row['name'].title()
                artist_name = row['artist'].title()

                st.markdown(f'###{ind}.  **{song_name}** by **{artist_name}')
                st.audio(row['spotify_preview_url'])
                st.write('----') 
























# # Get Recommendation Button
# if filtering_type == 'Content-based filtering':
#     if st.button('Get Recommendation'):
#         if ((songs_data['name'] == song_name) & (songs_data['artist']== artist_name)).any():
#             st.write ('Recommendations for ', f"**{song_name}**")
#             st.write(f' Recommendation for ', song_name)
#             recommendations = content_recommendation(song_name, artist_name, songs_data, transformed_data)  # data is cleaned_data, trnasformed_data is loaded npz data
#             # the top 10 songs (name, artist and spotify_url are "Returned" from the function and stored in recommendation variable)

#             # display Recommendation 
#             for ind, row in recommendations.iterrows():  # iterate through every row of the df which content_recommendation function is returning
#                 song_name = row['name'].title()
#                 artist_name = row['artist'].title()

#                 st.markdown(f'###{ind}.  **{song_name}** by **{artist_name}')
#                 st.audio(row['spotify_preview_url'])
#                 st.write('----')

#         else:
#             st.write (f'Sorry , we coudnt find {song_name} in  our database of songs')

# elif filtering_type == 'Collaborative-based filtering':
#     if st.button('Get Recommendation'):
#         if ((songs_data['name'] == song_name) & (songs_data['artist']== artist_name)).any():
#             st.write ('Recommendations for ', f"**{song_name}**")
#             st.write(f' Recommendation for ', song_name)
#             recommendations= collaborative_recommendation(song_name, artist_name, track_ids, filtered_data, interaction_matrix) 

#             # display Recommendation 
#             for ind, row in recommendations.iterrows():  # iterate through every row of the df which content_recommendation function is returning
#                 song_name = row['name'].title()
#                 artist_name = row['artist'].title()

#                 st.markdown(f'###{ind}.  **{song_name}** by **{artist_name}')
#                 st.audio(row['spotify_preview_url'])
#                 st.write('----')
#         else:
#             st.write (f'Sorry , we coudnt find {song_name} in  our database of songs')

# elif filtering_type == 'Hybrid Recommender System':
#     if st.button('Get Recommendation'):
#         if ((filtered_data['name'] == song_name) & (filtered_data['artist']== artist_name)).any():
#             st.write ('Recommendations for ', f"**{song_name}**")
#             st.write(f' Recommendation for ', song_name)

#             diversity = st.slider(label = "Diversity in Recommendations",
#                                   min_value = 1,
#                                   max_value =10,
#                                   value =5,
#                                   step =1 )
#             content_based_weight = 1 - (diversity/10) # to keep the values between 0 and 1

            
#             recommender = hrs(number_of_recommendations=5,
#                               weight_content_based=content_based_weight,
#                               )
#             recommendations = recommender.give_recommendations(song_name=song_name, artist_name=artist_name,
#                                                                song_data= filtered_data,                                                               transformed_matrix=transform_hybrid_data,
#                                                                track_ids=track_ids,
#                                                                interaction_matrix=interaction_matrix)

#             # display Recommendation 
#             for ind, row in recommendations.iterrows():  # iterate through every row of the df which content_recommendation function is returning
#                 song_name = row['name'].title()
#                 artist_name = row['artist'].title()

#                 st.markdown(f'###{ind}.  **{song_name}** by **{artist_name}')
#                 st.audio(row['spotify_preview_url'])
#                 st.write('----') 

#         else: 
#             filtering_type == 'Content_based_filtering'
#             # if st.button('Get Recommendation'):
#             #     if ((songs_data['name'] == song_name) & (songs_data['artist']== artist_name)).any():
#             #         st.write ('Recommendations for ', f"**{song_name}**")
#             #         st.write(f' Recommendation for ', song_name)
#             recommendations = content_recommendation(song_name, artist_name, songs_data, transformed_data)  # data is cleaned_data, trnasformed_data is loaded npz data
#                     # the top 10 songs (name, artist and spotify_url are "Returned" from the function and stored in recommendation variable)

#                     # display Recommendation 
#             for ind, row in recommendations.iterrows():  # iterate through every row of the df which content_recommendation function is returning
#                 song_name = row['name'].title()
#                 artist_name = row['artist'].title()

#                 st.markdown(f'###{ind}.  **{song_name}** by **{artist_name}')
#                 st.audio(row['spotify_preview_url'])
#                 st.write('----')

#     else:
#             st.write (f'Sorry , we coudnt find {song_name} in  our database of songs')

# else : 
#     st.write (f'Sorry , we coudnt find {song_name} in  our database of songs to recommend')




