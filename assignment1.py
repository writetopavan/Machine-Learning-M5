#
# TOOD: Import whatever needs to be imported to make this work
#
# .. your code here ..
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
matplotlib.style.use('ggplot') # Look Pretty


#
# TODO: To procure the dataset, follow these steps:
# 1. Navigate to: https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2
# 2. In the 'Primary Type' column, click on the 'Menu' button next to the info button,
#    and select 'Filter This Column'. It might take a second for the filter option to
#    show up, since it has to load the entire list first.
# 3. Scroll down to 'GAMBLING'
# 4. Click the light blue 'Export' button next to the 'Filter' button, and select 'Download As CSV'

df=pd.read_csv('D:\learning\DAT210x-master\Module5\Crimes_-_2001_to_present.csv')

def doKMeans(df):
  #
  # INFO: Plot your data with a '.' marker, with 0.3 alpha at the Longitude,
  # and Latitude locations in your dataset. Longitude = x, Latitude = y
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(df.Longitude, df.Latitude, marker='.', alpha=0.3)

  #
  # TODO: Filter df so that you're only looking at Longitude and Latitude,
  # since the remaining columns aren't really applicable for this purpose.
  #
  # .. your code here ..
ab=df[['Latitude','Longitude']]
ab=ab.dropna()

  #
  # TODO: Use K-Means to try and find seven cluster centers in this df.
  #
  # .. your code here ..

kmeans = KMeans(n_clusters=7)
kmeans.fit(ab)
KMeans(copy_x=True, init='k-means++', max_iter=300, n_clusters=7, n_init=10,
    n_jobs=1, precompute_distances='auto', random_state=None, tol=0.0001,
    verbose=0)

#labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_
T=pd.DataFrame(centroids)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component2', y='component1', marker='o', c='r', alpha=0.5, linewidths=3, s=169)
plt.show()
  #
  # INFO: Print and plot the centroids...
  #centroids = kmeans_model.cluster_centers_




#
# TODO: Load your dataset after importing Pandas
#
# .. your code here ..


#
# TODO: Drop any ROWs with nans in them
#
# .. your code here ..

cd=df.dropna()
#
# TODO: Print out the dtypes of your dset
#
# .. your code here ..


#
# Coerce the 'Date' feature (which is currently a string object) into real date,
# and confirm by re-printing the dtypes. NOTE: This is a slow process...
#
# .. your code here ..
cd.Date = pd.to_datetime(cd.Date, errors='coerce')

cd.dtypes
# INFO: Print & Plot your data
doKMeans(cd)


#
# TODO: Filter out the data so that it only contains samples that have
# a Date > '2011-01-01', using indexing. Then, in a new figure, plot the
# crime incidents, as well as a new K-Means run's centroids.
#
# .. your code here ..
EF=cd[cd.Date>'2011-01-01']
GH=EF[['Latitude','Longitude']]
kmeans = KMeans(n_clusters=7)
kmeans.fit(GH)
KMeans(copy_x=True, init='k-means++', max_iter=300, n_clusters=7, n_init=10,
    n_jobs=1, precompute_distances='auto', random_state=None, tol=0.0001,
    verbose=0)

#labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_
centroids
T=pd.DataFrame(centroids)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component2', y='component1', marker='o', c='r', alpha=0.5, linewidths=3, s=169)
plt.show()

# INFO: Print & Plot your data
doKMeans(GH)
plt.show()


