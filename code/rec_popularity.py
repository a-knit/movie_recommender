import pandas as pd
import graphlab as gl
import numpy as np
import operator

def manip_sample_sub_format(sample_sub):
    """
    Return SFrame based on sample sub, with movid id pulled into separate format for consistency
    with the data used for model fitting.
    """
    sample_sub['movie'] = sample_sub.id.apply(lambda x: int(x.split('_')[1]))
    for_prediction = gl.SFrame(sample_sub)
    return for_prediction

def load_user_data(ratings=None, SFrame=True):
    users = pd.read_table('/Users/datascientist/Desktop/recommender-case-study/data/users.dat', sep='::', header=None, names=['user', 'gender', 'age-group', 'occupation', 'zip'])
    gender_dummies = pd.get_dummies(users.gender, prefix='gender')
    users = users.merge(gender_dummies, left_index=True, right_index=True)
    users.drop('gender', axis=1, inplace=True)
    users.drop('gender_M', axis=1, inplace=True)
    users.drop('zip', axis=1, inplace=True)
    users = add_rating_number_to_users(ratings, users)
    users.columns = ['user', 'age-group', 'occupation', 'gender_F', 'count']
    return gl.SFrame(users)

def load_genre_data():
    movies = pd.read_table('data/movies.dat', sep='::', header=None, names=['movie', 'title', 'genres'])
    movies['year'] = movies['title'].map(lambda x: int(x[-5:-1]))
    movies['title'] = movies['title'].map(lambda x: x[:-7])
    movies['genres'] = movies['genres'].map(lambda x: x.split('|'))
    
    genres = []
    for row in movies['genres']:
        genres += row
    genres = set(genres)
    genres_d = {}
    for i, genre in enumerate(genres):
        genres_d[genre] = i
    genre_mat = np.zeros((len(movies), len(genres)))
    i = 0
    for movie_genres in movies['genres']:
        for genre in movie_genres:
            genre_mat[i][genres_d[genre]] = 1
        i += 1  
    sort = sorted(genres_d.items(), key=operator.itemgetter(1))
    sort = [x[0] for x in sort]
    genres = pd.DataFrame(genre_mat, columns=sort)
    genres = pd.merge(movies, genres, left_index=True, right_index=True)
    return gl.SFrame(genres.drop(['title', 'genres'], axis=1))

def add_rating_number_to_users(ratings, users):
    num_ratings_per_user = ratings.groupby('user').count()
    users = pd.merge(users, num_ratings_per_user, left_on='user', right_index=True, how='outer')
    return users.drop(['movie', 'rating'], axis=1).fillna(0)

def pick_model(users, cold_model, hot_model, thresh):
    cold = pd.read_csv(cold_model)
    hot = pd.read_csv(hot_model)
    users = pd.DataFrame(users)
    hot_ratings = np.array(hot['rating'])
    cold_ratings = np.array(cold['rating'])
    merged = pd.merge(cold, users, how='left', left_on='user', right_on='user')
    mask = np.array(merged['count'] >= 20)
    np.putmask(cold_ratings, mask, hot_ratings)
    final_pred = cold.copy()
    final_pred['rating'] = cold_ratings
    final_pred.to_csv('data/chosen_predictions.csv', index=False)

if __name__ == "__main__":
    sample_sub_fname = "data/sample_submission.csv"
    ratings_data_fname = "data/training_ratings.csv"
    cold_fname = "data/predict_ratings_popularity.csv"
    hot_fname = "data/predict_rankings_factorization.csv"

    ratings = pd.read_csv(ratings_data_fname)
    sample_cold = pd.read_csv(sample_sub_fname)
    sample_hot = sample_cold.copy()
    for_prediction = manip_sample_sub_format(sample_cold)
    users = load_user_data(ratings=ratings)
    movies = load_genre_data()
    ratings = gl.SFrame(ratings_data_fname)
    cold_engine = gl.popularity_recommender.create(   observation_data=ratings, 
                                                        user_id="user", 
                                                        item_id="movie", 
                                                        target='rating',
                                                        user_data=users,
                                                        item_data=movies)
    
    sample_cold.rating = cold_engine.predict(for_prediction)
    sample_cold.drop('movie', inplace=True, axis=1)
    sample_cold.to_csv(cold_fname, index=False)

    for_prediction = manip_sample_sub_format(sample_hot)

    hot_engine = gl.recommender.factorization_recommender.create(   
                                                        observation_data=ratings, 
                                                        user_id="user", 
                                                        item_id="movie", 
                                                        target='rating',
                                                        )
    
    sample_hot.rating = hot_engine.predict(for_prediction)
    sample_hot.drop('movie', inplace=True, axis=1)
    sample_hot.to_csv(hot_fname, index=False)
    pick_model(users.to_dataframe(), cold_fname, hot_fname, 20)