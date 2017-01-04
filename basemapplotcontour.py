import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as pl
import numpy as np
import seaborn as sns
import zipfile


pl.figure(figsize=(20,5))
map_extent = [-122.53, 37.68, -122.35, 37.83]
m = Basemap(llcrnrlon=map_extent[0], llcrnrlat=map_extent[1],
             urcrnrlon=map_extent[2], urcrnrlat=map_extent[3],projection='lcc',resolution='f', epsg=4269)
m.drawcoastlines()
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color = 'gray')

longitudes = np.arange(-122.53, -122.35, .02)
latitudes = np.arange(37.68, 37.83, .02)

m.drawparallels(latitudes,labels=[1,1,0,0])
m.drawmeridians(longitudes,labels=[0,0,0,1])

asp = 1

lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]

z = zipfile.ZipFile('../input/train.csv.zip')
train = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'])

#print(train.head())

#Get rid of the bad lat/longs
train['Xok'] = train[train.X<-121].X
train['Yok'] = train[train.Y<40].Y
train = train.dropna()
train_cat = train[train.Category == 'ASSAULT'] 
train_cat = train_cat[1:3000] #Can't use all the data and complete within 600 sec :(

ax = sns.kdeplot(train_cat.Xok, train_cat.Yok,n_levels=20, clip=clipsize, aspect=1/asp)
ax= pl.gca()

pl.show()



def categoryCrime(data):
    #print(data)
    categories = {}
    print(data.columns)
    category = data['Category']
    for cate in category:
        if cate in categories:
            categories[cate] += 1
        else:
            categories[cate] = 1
            
            
            
            





