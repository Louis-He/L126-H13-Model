from mpl_toolkits.basemap import Basemap, cm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import color

import h5py as h5
import numpy as np

LON = 121.44
LAT = 31.19

def readInitialPM2_5Field(hour):
    field = np.zeros((161,281))
    if hour == '00':
        file = h5.File('air.hdf5', 'r')
        beforeProcess = file['pm25'][:]
        for i in range(161):
            field[i][:] = beforeProcess[160 - i][:]
    else:
        file = h5.File('output/L126_H13_' + str(hour) + '.hdf5', 'r')
        field = file['pm25'][:]
        field = np.ma.array(field)

    time = file['time']
    time = np.array(time)
    return {'pm25': field, 'time': time}

def plot(title, timeSeries, pm25Series):
    plt.figure(figsize=(12, 5))
    plt.plot(timeSeries,  pm25Series)
    plt.ylabel("PM2.5 (ug/m**3)")
    plt.xlabel("Forecast hour")
    plt.title("Forecast of concentration of PM2.5 - " + title + "\nL126_H13 Regional Air Quality Model [In trial] @Louis-He")
    plt.xticks(rotation = 90)

    plt.savefig('output/timeSeries.png')

    
timeSeries = []
pm25Series = []

for i in range(96):
    if i < 10:
        i = '0' + str(i)
    else:
        i = str(i)

    gridLon = int(round((LON - 70) * 4))
    gridLat = int(round((LAT - 15) * 4))

    Field = readInitialPM2_5Field(i)

    timeSeries.append(i)
    pm25Series.append(Field['pm25'][gridLat][gridLon])

plot(Field['time'] + ' - ' + str(LON) + 'E, ' + str(LAT) + 'N', timeSeries, pm25Series)
