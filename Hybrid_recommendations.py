import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

class HybridRecommenderSystem:

    def __init__(self,
                 number_of_recommendations: int,
                 weight_content_based: float,
                 ):
        
        self.number_of_recommendation = number_of_recommendations
        self.weight_content_based = weight_content_based
        self.weight_collaborative_based = 1- weight_content_based
      
    
    def calculate_content_based_similarities(self,song_name,artist_name, song_data, transformed_matrix):
        # Filter out song from data
        song_row = song_data.loc[(song_data['name'] == song_name) & (song_data['artist'] == artist_name)]
        # get index of song
        song_index = song_row.index[0]
        #generate input vector
        input_vector  = transformed_matrix[song_index].reshape(1,-1)
        #calculate similarity scores
        content_similarity_scores = cosine_similarity(input_vector, transformed_matrix)
        return content_similarity_scores
    
    def calculate_colaborative_based_similarities(self,song_name,artist_name, track_ids, song_data, interaction_matrix ):
        #fetch the row from song_data
        song_row = song_data.loc[(song_data['name']== song_name) & (song_data['artist']== artist_name)]
        #track_id of input song : gets only the track_id value without the inverted commas ' TV123494959'
        input_track_id = song_row['track_id'].values.item()
        #get the index of the track_id: this index is got from user listneing music info db
        ind = np.where(track_ids==input_track_id)[0].item()
        input_array = interaction_matrix[ind]  #interaction_matrix.iloc[index_value]
        collaborative_similarity_scores = cosine_similarity(input_array, interaction_matrix)
        return collaborative_similarity_scores
    
    def normalize_similarities(self, similarity_scores):  # Normalising so that the Cosine similairty is same for content based and collaborative based
        # so that when we multiply with wieghts, equal weightage is obtained.
        minimum = np.min(similarity_scores)
        maximum = np.max(similarity_scores)
        normalized_scores = (similarity_scores - minimum) / (maximum - minimum)
        return normalized_scores
    
    def weighted_combinations(self,content_based_scores, collaborative_based_scores):
        weighted_scores = (self.weight_content_based*content_based_scores) + (self.weight_collaborative_based * collaborative_based_scores)
        return weighted_scores
    
    def give_recommendations(self, song_name, artist_name, song_data, track_ids, transformed_matrix, interaction_matrix ):
        #calculate content based similarities # The function returns the cosine similarities and stored in the 'content_based_similarities' variable
        # the atrritubutes passed to the calculate content /collaborative functions are obtained from the recommender object created in app.py
        # for eg " transformed matrix = transformed matrix" here the RHS transformed_matrix is the parameter passed to recommender object( tranaformed matrix = transformed_data) and transformed_data is the attribute - thr actual value 
        # recommender = hrs(parameters)
        content_based_similarities = self.calculate_content_based_similarities(song_name=song_name,  # song_name1 = attribute(Love Story) passed to song_name while creating the constructor
                                                                               artist_name=artist_name,
                                                                               song_data = song_data,
                                                                               transformed_matrix=transformed_matrix)
        # eg here the tranformed_matrix on rhs is attribute which is a parameter in the HRS() object in the app.py . the attribute is transformed_data 
        
        #calculate collaboartive based similarities
        collaborative_based_similarities = self.calculate_colaborative_based_similarities(song_name = song_name,
                                                                                          artist_name=artist_name,
                                                                                          track_ids= track_ids,
                                                                                          song_data=song_data,
                                                                                          interaction_matrix = interaction_matrix)
        #normalise the similarities
        normalize_content_based_similarities = self.normalize_similarities(content_based_similarities)
        normalize_collaborative_based_similarities = self.normalize_similarities(collaborative_based_similarities)

        #once Normalised multiplying it with its weights
        
        weighted_scores = self.weighted_combinations(content_based_scores= normalize_content_based_similarities, collaborative_based_scores = normalize_collaborative_based_similarities)

         # index values of recommendations : np.argsort, sort and return the indexes based on values
        recommendation_indices = np.argsort(weighted_scores.ravel())[-self.number_of_recommendation-1:][::-1] 
        
        # get top k recommendations
        recommendation_track_ids = track_ids[recommendation_indices]
       
        # get top scores
        top_scores = np.sort(weighted_scores.ravel())[-self.number_of_recommendation-1:][::-1]
        
        # get the songs from data and print
        scores_df = pd.DataFrame({"track_id":recommendation_track_ids.tolist(),
                                "score":top_scores})
        top_k_songs = (
                        song_data
                        .loc[song_data["track_id"].isin(recommendation_track_ids)]
                        .merge(scores_df,on="track_id")
                        .sort_values(by="score",ascending=False)
                        .drop(columns=["track_id","score"])
                        .reset_index(drop=True)
                        )
        
        return top_k_songs[['name','artist','spotify_preview_url']]
    
# if __name__ == "__main__" : 
#     #load transformed data ( the filtered data 50k songs reduced to 30k dataset base don songs in User interaction Dataset, that data is transformed used the column transformer)
#     transformed_data = load_npz("D:/spotify_recommender/data/transformed_hybrid_data.npz")

#     # load interaction matrix 
#     interaction_matrix = load_npz("D:/spotify_recommender/data/interaction_matrix.npz")

#     #load track_ids
#     track_ids = np.load("D:/spotify_recommender/data/track_ids.npy", allow_pickle=True)

#     #load the song_data main dataset
#     song_data = pd.read_csv("D:/spotify_recommender/data/collab_filtered_data1.csv")

#     #checking to see if it works by creating an instance
#     hybrid_recommender = HybridRecommenderSystem(song_name="take time",  # this value is stored in self.song_name1 variable and passed to the content based filtering function
#                                                  artist_name= "the books",
#                                                  number_of_recommendations=5,
#                                                  weight_collaborative_based=0.7,
#                                                  weight_content_based=0.3,
#                                                  song_data=song_data,
#                                                  transformed_matrix=transformed_data,
#                                                  interaction_matrix=interaction_matrix,
#                                                  track_ids=track_ids)
    
#     recommendations = hybrid_recommender.give_recommendations()
#     print(recommendations)


########################################################################################################################3
# class Car:
#     def __init__(self, brand: str, model,year:int):
#         self.b = brand.lower()
#         self.c = model
#         self.y = year

#     def car_brand(self, brand):
        
#         brandy = brand
#         return brandy
    
#     def give_recommendations(self):
#         x = self.car_brand(brand = self.b)
#         return x
    
# mycar= Car('TOYOTA', 'RAV', 10)
# x = mycar.car_brand("TOYOTA")
# print(x)

# y = mycar.give_recommendations()
# print(y)
# Result of x : TOYOTA
# Result of y: toyota

# car_brand function needs an argument 'brand'. this argument is passed when we call give recommendation and give_recommendation calls the function car_brand. 
# The argument which car_brand needs is obtained from the constructor when we create the object for first time

    






