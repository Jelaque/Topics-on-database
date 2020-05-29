from ml100k import recommenderMl100k
import time as tm
from distances import recommender

s = recommender(0)
s.readMovies()

'''
s = recommender(0,k=3,metric='manhattan')
s.readBooks()
#print(s.jaccard(s.data['Stephen'],s.data['Amy']))
print(s.ProjectedRanting('Patrick C','Scarface'))
'''
'''
r = recommenderMl100k(0,metric='cosine')
r.loadMovieLens('../datasets/ml-100k/')
#print(r.cosine(r.data['278833"'],r.data['278858"']))
#print(r.jaccard(r.data['278804'],r.data['211']))
print(r.computeNearestNeighbor("100"))
'''