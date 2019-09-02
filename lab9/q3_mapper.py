import sys

# input comes from STDIN (standard input)
movieid_name={}
for line in sys.stdin:
    line = line.strip()
    line = line.split(",")
    
    movie_id_map=line[0]
    movie_name=line[1]
    genre=line[2]
    user_id = line[3]
    movie_id= line[4]
    rating=line[5]
    
#        