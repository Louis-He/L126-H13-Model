from mpl_toolkits.basemap import Basemap, cm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import color

import h5py as h5
import numpy as np
import sys

def readInitialPM2_5Field(hour, GFSheadHour):
    field = np.zeros((161,281))
    if hour == '00':
        file = h5.File('air.hdf5', 'r')
        beforeProcess = file['pm25'][:]
        for i in range(161):
            field[i][:] = beforeProcess[160 - i][:]
    else:
        file = h5.File('output/' + str(GFSheadHour) + '/L126_H13_' + str(hour) + '.hdf5', 'r')
        field = file['pm25'][:]
        field = np.ma.array(field)

    time = file['time']
    time = np.array(time)
    return {'pm25': field, 'time': time}

def plot(title, timeSeries, pm25Series, GFSheadHour):
    plt.figure(figsize=(18, 5))
    plt.plot(timeSeries,  pm25Series)
    plt.ylabel("PM2.5 (ug/m**3)")
    plt.xlabel("Forecast hour")
    plt.title("Forecast of concentration of PM2.5 - " + title + "\nL126_H13 Regional Air Quality Model [Experimental beta 1.0.0] @Louis-He")
    plt.xticks(rotation = 90)

    plt.savefig('output/' + GFSheadHour + '/' + title + '.png')
    plt.clf()
    plt.close()

def saveForecastFile(title, timeSeries, pm25Series, GFSheadHour):
    f = open('output/' + GFSheadHour + '/' + title + '.csv', 'w+')
    f.write('L126-H13 Model Forecast\n')
    f.write(title[:title.find(' - ')] + '\n')
    f.write(title[title.find(' - ') + len(' - '):] + '\n')
    f.write('\n')
    f.write('Hour, Concentration\n')
    for i in range(len(timeSeries)):
        f.write(str(int(timeSeries[i])) + ', ' + str(pm25Series[i]) + '\n')
    f.close()

def singlePointForecast(lon, lat, GFSheadHour):
    timeSeries = []
    pm25Series = []
    for i in range(97):
        if i < 10:
            i = '0' + str(i)
        else:
            i = str(i)

        gridLon = int(round((lon - 70) * 4))
        gridLat = int(round((lat - 15) * 4))

        Field = readInitialPM2_5Field(i, GFSheadHour)

        timeSeries.append(i)
        pm25Series.append(Field['pm25'][gridLat][gridLon])

    plot(Field['time'] + ' - ' + str(lon) + 'E, ' + str(lat) + 'N', timeSeries, pm25Series, GFSheadHour)
    saveForecastFile(Field['time'] + ' - ' + str(lon) + 'E, ' + str(lat) + 'N', timeSeries, pm25Series, GFSheadHour)

nargs = len(sys.argv)
skip = False
for i in range(1,nargs):
   if not skip:
      arg = sys.argv[i]
      print ("INFO: processing" + arg)
      if arg == "--GFSheadHour":
         if i != nargs-1:
            GFSheadHour = sys.argv[i+1]
            skip=True
      else:
         print ("ERR: unknown arg:" + arg)
   else:
      skip=False

singlePointForecast(121.41, 31.17, GFSheadHour) # 徐汇
singlePointForecast(116.37, 39.87, GFSheadHour) # 1011
singlePointForecast(120.21, 30.21, GFSheadHour) # 1223