# This shows how to read the text representing a map of Chicago in numpy, and put it on a plot in matplotlib.
# This example doesn't make it easy for you to put other data in lat/lon coordinates on the plot.
# Hopefully someone else can add an example showing how to do that? You'll need to know the bounding box of this map:
#    ll.lon     ll.lat   ur.lon     ur.lat
#    -122.52469 37.69862 -122.33663 37.82986



import pandas as pd
import zipfile
import numpy as np
import matplotlib.pyplot as pl

z = zipfile.ZipFile('train.csv.zip')
print(z.namelist())

train = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'])

SF_map= np.loadtxt("../input/sf_map_copyright_openstreetmap_contributors.txt")
# Supplied map bounding box:
#    ll.lon     ll.lat   ur.lon     ur.lat
#    -122.52469 37.69862 -122.33663 37.82986
lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
asp = SF_map.shape[0] * 1.0 / SF_map.shape[1]
fig = pl.figure(figsize=(16,16))
pl.imshow(SF_map,cmap='bone',extent=lon_lat_box,aspect=1/asp)
ax= pl.gca()

train = train[train['Y']<40]
train_cat = train[train['Category']=="VANDALISM"]
train_cat.plot(x = 'X',y = 'Y',ax=ax,kind='scatter',marker='o',s=2,color='red',alpha=0.04)

pl.savefig('TotalCrimeonMap.png')
