

#Reducer.py
import sys
import operator



movieid_name={}
movieid_rating={}
userid_rating={}
genre_rating={}
ave_movie_rating={}
ave_user_rating={}
ave_genre_rating={}

for line in sys.stdin:
    data = line.strip().split('\t')
    
    if len(data) != 6:
        # Something has gone wrong. Skip this line.
        continue
    movie_id_map,movie_name,genre,user_id,movie_id,rating= data
    
    if movie_id_map in movieid_name:
        movieid_name[movie_id_map].append(int(movie_name))
    else:
        movieid_name[movie_id_map] = []
        movieid_name[movie_id_map].append(int(movie_name))
        
    if movie_id in movieid_rating:
        movieid_rating[movie_id].append(int(rating))
    else:
        movieid_rating[movie_id] = []
        movieid_rating[movie_id].append(int(rating))
        
    if user_id in userid_rating:
        userid_rating[user_id].append(int(rating))
    else:
        userid_rating[user_id] = []
        userid_rating[user_id].append(int(rating))
        
    if genre in genre_rating:
        genre_rating[genre].append(int(rating))
    else:
        genre_rating[genre] = []
        genre_rating[genre].append(int(rating))
   
    
              
for movie_id in movieid_rating.keys():
    ave_movie_rating[movieid_name[movie_id]].append(sum(movieid_rating[movie_id])*1.0 / len(movieid_rating[movie_id]))
print(max(ave_movie_rating.iteritems(), key=operator.itemgetter(1))[0])

for user_id in userid_rating.keys():
    if(len(userid_rating[user_id]>40)):
        ave_user_rating[user_id].append(sum(userid_rating[user_id])*1.0 / len(userid_rating[user_id]))   
    
print(min(ave_user_rating.iteritems(), key=operator.itemgetter(1))[0])

for genre in genre_rating.keys():
    ave_genre_rating[genre_rating[genre]].append(sum(genre_rating[genre])*1.0 / len(genre_rating[genre]))
    
print(max(ave_genre_rating[genre].iteritems(), key=operator.itemgetter(1))[0])








    