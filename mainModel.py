from mpl_toolkits.basemap import Basemap, cm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import color

import pygrib
from netCDF4 import Dataset
import numpy as np
import math
import h5py as h5
import time as times
import datetime
import os
import sys

month = datetime.datetime.now().month
PI = 3.1415926535857932
EARTHRADIUS = 6371000.0 # unit: m

center = 0.92 # 1
cocenter = 0.02  # 4
subcenter = 0.000  # 8
subouter = 0.000  # 8
outer = 0  # 4

######################### INPUT HELPER FUNCTIONS BEGIN ###########################

# return source of PM2.5 from
# 15 ~ 55: 15 ~ 55
# 70 ~ 140: 70 ~ 140
# shape: 161 * 281
def readinventoryPM2_5(month, fct):
    dataset = Dataset('sourceInventory/MICS_Asia_PM2.5_2010_0.25x0.25.nc')
    lons = dataset.variables['lon'] # 40.125 ~ 179.875
    lats = dataset.variables['lat'] # -20.125 ~ 89.875

    source_INDUSTRY = dataset.variables['PM2.5_INDUSTRY'][month - 1]
    source_POWER = dataset.variables['PM2.5_POWER'][month - 1]

    if fct < 5:
        source_TRANSPORT = dataset.variables['PM2.5_TRANSPORT'][month - 1] * 0.7
        source_RESIDENTIAL = dataset.variables['PM2.5_RESIDENTIAL'][month - 1] * 0.75
    elif fct < 6:
        source_TRANSPORT = dataset.variables['PM2.5_TRANSPORT'][month - 1] * 0.85
        source_RESIDENTIAL = dataset.variables['PM2.5_RESIDENTIAL'][month - 1] * 0.85
    elif fct < 10:
        source_TRANSPORT = dataset.variables['PM2.5_TRANSPORT'][month - 1]
        source_RESIDENTIAL = dataset.variables['PM2.5_RESIDENTIAL'][month - 1]
    elif fct < 17:
        source_TRANSPORT = dataset.variables['PM2.5_TRANSPORT'][month - 1] * 0.95
        source_RESIDENTIAL = dataset.variables['PM2.5_RESIDENTIAL'][month - 1]
    elif fct < 21:
        source_TRANSPORT = dataset.variables['PM2.5_TRANSPORT'][month - 1]
        source_RESIDENTIAL = dataset.variables['PM2.5_RESIDENTIAL'][month - 1]
    elif fct < 23:
        source_TRANSPORT = dataset.variables['PM2.5_TRANSPORT'][month - 1] * 0.85
        source_RESIDENTIAL = dataset.variables['PM2.5_RESIDENTIAL'][month - 1]
    elif fct < 25:
        source_TRANSPORT = dataset.variables['PM2.5_TRANSPORT'][month - 1] * 0.75
        source_RESIDENTIAL = dataset.variables['PM2.5_RESIDENTIAL'][month - 1] * 0.85
    else:
        source_TRANSPORT = dataset.variables['PM2.5_TRANSPORT'][month - 1]
        source_RESIDENTIAL = dataset.variables['PM2.5_RESIDENTIAL'][month - 1] * 0.75

    source = source_RESIDENTIAL + source_INDUSTRY + source_POWER + source_TRANSPORT
    source = source[140:302, 119:401] # 14.875~55.125, 69.875 ~ 140.125

    newSource = np.zeros((161, 281)) # 161 rows, 281 cols
    for i in range(161):
        for j in range(281):
            newSource[i][j] = (source[i][j] + source[i][j + 1] + \
                              source[i + 1][j] + source[i + 1][j + 1]) / 4.0
    return newSource

# return Initial pollutant field of PM2.5
# 15 ~ 55: 15 ~ 55
# 70 ~ 140: 70 ~ 140
# shape: 161 * 281
def readInitialPM2_5Field(hour, GFSheadHour):
    field = np.zeros((161,281))
    if hour == '00':
        file = h5.File('air.hdf5', 'r')
        beforeProcess = file['pm25'][:]
        for i in range(161):
            field[i][:] = beforeProcess[160 - i][:]
    else:
        file = h5.File('output/' + GFSheadHour + '/L126_H13_' + str(hour) + '.hdf5', 'r')
        field = file['pm25'][:]
        field = np.ma.array(field)

    time = file['time']
    time = np.array(time)
    return {'pm25': field, 'time': time}

######################### INPUT HELPER FUNCTIONS END ###########################

######################### ANALYSIS HELPER FUNCTIONS BEGIN ###########################

# shape: 161 * 281
def preProcess(pm25Grid, grbs, pm2_5Inventory, GFSBeforeHeadStr):
    land = grbs.select(name='Land-sea coverage (nearest neighbor) [land=1,sea=0]')[0]
    land = land.values

    try:
        hdf5FilePath = 'output/' + GFSBeforeHeadStr + '/L126_H13_06.hdf5'
        ForecastFile = h5.File(hdf5FilePath)
        ForecastField = ForecastFile['pm25'][:]
        for i in range(161):
            for j in range(281):
                if pm25Grid[i][j] == 0:
                    pm25Grid[i][j] = ForecastField[i][j]

                if (pm25Grid[i][j] < 5):
                    pm25Grid[i][j] = 5

    except:
        for i in range(161):
            for j in range(281):
                if pm25Grid[i][j] == 0:
                    pm25Grid[i][j] = 5

                if pm2_5Inventory[i][j] / 3 > 5 and pm25Grid[i][j] == 5:
                    pm25Grid[i][j] = pm2_5Inventory[i][j] / 3
                elif(land[i][j] == 1 and pm25Grid[i][j] == 5):
                    pm25Grid[i][j] = 20
                elif(land[i][j] == 0 and pm25Grid[i][j] == 5):
                    pm25Grid[i][j] = 5

    return pm25Grid

# shape: 161 * 281
def windComponent(pm25Grid, grbs, grbs2):
    ITERATIVE_TIMES = 3

    # U component of wind
    # v component of wind

    # this hour
    # wind10m_u = grbs.select(name='10 metre U wind component')[0]
    wind10m_u = grbs.select(name='U component of wind')[35]
    wind10m_u = np.array(wind10m_u.values)
    # wind10m_v = grbs.select(name='10 metre V wind component')[0]
    wind10m_v = grbs.select(name='V component of wind')[35]
    wind10m_v = np.array(wind10m_v.values)

    # next hour
    # wind10m_u2 = grbs2.select(name='10 metre U wind component')[0]
    wind10m_u2 = grbs2.select(name='U component of wind')[35]
    wind10m_u2 = np.array(wind10m_u2.values)
    # wind10m_v2 = grbs2.select(name='10 metre V wind component')[0]
    wind10m_v2 = grbs2.select(name='V component of wind')[35]
    wind10m_v2 = np.array(wind10m_v2.values)

    # windSpeed = (wind10m_u ** 2 + wind10m_v ** 2) ** (1.0 / 2.0)
    # windSpeed2 = (wind10m_u2 ** 2 + wind10m_v2 ** 2) ** (1.0 / 2.0)

    # be AWARE OF CONVENTION!!!!!
    wind10m_u_process = np.zeros((161,281))
    wind10m_v_process = np.zeros((161,281))


    # calculate wind
    tmpForecastWindComponent = pm25Grid

    for k in range(ITERATIVE_TIMES):

        forecastWindComponent = np.zeros((161, 281))

        for i in range(len(wind10m_u)):
            for j in range(len(wind10m_u[0])):

                # average wind field
                initialPortion = (k + 1.0) / (ITERATIVE_TIMES + 1.0)
                nextHourPortion = 1.0 - initialPortion

                if i >= 3 and i <= 157 and j >= 3 and j <= 277:
                    for m in range(-3, 4):
                        for l in range(-3, 4):
                            if m == -3 or l == -3 or m == 4 or l == 4:
                                wind10m_u_process[i][j] += (wind10m_u[i + m][j + l] * 0.0304 * initialPortion + wind10m_u2[i + m][j + l] * 0.0304 * nextHourPortion) / 2.0
                                wind10m_v_process[i][j] += (wind10m_v[i + m][j + l] * 0.0304 * initialPortion + wind10m_v2[i + m][j + l] * 0.0304 * nextHourPortion) / 2.0
                            else:
                                wind10m_u_process[i][j] += (wind10m_u[i + m][j + l] * 0.01 * initialPortion + wind10m_u2[i + m][j + l] * 0.01 * nextHourPortion) / 2.0
                                wind10m_v_process[i][j] += (wind10m_v[i + m][j + l] * 0.01 * initialPortion + wind10m_v2[i + m][j + l] * 0.01 * nextHourPortion) / 2.0
                else:
                    wind10m_u_process[i][j] += (wind10m_u[i][j] * initialPortion + wind10m_u2[i][j] * nextHourPortion) / 2.0
                    wind10m_v_process[i][j] += (wind10m_v[i][j] * initialPortion + wind10m_v2[i][j] * nextHourPortion) / 2.0

                tmpLatitude = i / 4.0 + 15
                earthCircumfirance = 2 * PI * math.sin(((90 - tmpLatitude) / 180.0) * PI) * EARTHRADIUS
                horizontalDistance = wind10m_u_process[i][j] * 60.0 * 60 / ITERATIVE_TIMES  # unit: m
                verticalDistance = wind10m_v_process[i][j] * 60.0 * 60 / ITERATIVE_TIMES  # unit: m

                horizontalGrid = horizontalDistance / ((2.0 * PI * EARTHRADIUS) / 360.0) * 4.0
                verticalGrid = verticalDistance / (earthCircumfirance / 360.0) * 4.0

                newHorizontalCenter = float(j + horizontalGrid) # lon grid
                newVerticalCenter = float(i + verticalGrid) # lat grid


                # upper
                upper = newVerticalCenter - 0.5

                # lower
                lower = newVerticalCenter + 0.5

                # left
                left = newHorizontalCenter - 0.5

                # right
                right = newHorizontalCenter + 0.5

                horizontalCentralLine = math.floor(newHorizontalCenter) + 0.5
                verticalCentralLine = math.floor(newVerticalCenter) + 0.5

                # LEFT UPPER grid
                if(isInGrid(math.floor(newVerticalCenter), math.floor(newHorizontalCenter))):

                    leftUpperPortion = (horizontalCentralLine - left) * (verticalCentralLine - upper)

                    forecastWindComponent[math.floor(newVerticalCenter)][math.floor(newHorizontalCenter)] += \
                        tmpForecastWindComponent[i][j] * leftUpperPortion


                # LEFT BOTTOM grid
                if(isInGrid(math.floor(newVerticalCenter) + 1, math.floor(newHorizontalCenter))):

                    leftBottomPortion = (horizontalCentralLine - left) * (lower - verticalCentralLine)

                    forecastWindComponent[math.floor(newVerticalCenter) + 1][math.floor(newHorizontalCenter)] += \
                        tmpForecastWindComponent[i][j] * leftBottomPortion


                # RIGHT UPPER grid
                if (isInGrid(math.floor(newVerticalCenter), math.floor(newHorizontalCenter) + 1)):

                    rightUpperPortion = (right - horizontalCentralLine) * (verticalCentralLine - upper)

                    forecastWindComponent[math.floor(newVerticalCenter)][math.floor(newHorizontalCenter) + 1] += \
                        tmpForecastWindComponent[i][j] * rightUpperPortion


                # RIGHT BOTTOM grid
                if(isInGrid(math.floor(newVerticalCenter) + 1, math.floor(newHorizontalCenter) + 1)):

                    rightBottomPortion = (right - horizontalCentralLine) * (lower - verticalCentralLine)

                    forecastWindComponent[math.floor(newVerticalCenter) + 1][math.floor(newHorizontalCenter) + 1] += \
                        tmpForecastWindComponent[i][j] * rightBottomPortion


        tmpForecastWindComponent = forecastWindComponent

    newforecastWindComponent = np.zeros((161, 281))
    for i in range(1, len(wind10m_u) - 1):
        for j in range(1, len(wind10m_u[0]) - 1):
            newforecastWindComponent[i][j] += forecastWindComponent[i - 1][j - 1] * 0.025
            newforecastWindComponent[i][j] += forecastWindComponent[i - 1][j] * 0.1
            newforecastWindComponent[i][j] += forecastWindComponent[i - 1][j + 1] * 0.025
            newforecastWindComponent[i][j] += forecastWindComponent[i][j - 1] * 0.1
            newforecastWindComponent[i][j] += forecastWindComponent[i][j] * 0.5
            newforecastWindComponent[i][j] += forecastWindComponent[i][j + 1] * 0.1
            newforecastWindComponent[i][j] += forecastWindComponent[i + 1][j - 1] * 0.025
            newforecastWindComponent[i][j] += forecastWindComponent[i + 1][j] * 0.1
            newforecastWindComponent[i][j] += forecastWindComponent[i + 1][j + 1] * 0.025

    return {"windComponent": newforecastWindComponent, "windu": wind10m_u, "windv": wind10m_v}

# shape: 161 * 281
def newPollutantFromSource(grbs, sourceInventory):
    # visibility = grbs.select(name='Visibility')[0]
    sourceContribution = np.zeros((161,281))

    for i in range(161):
        for j in range(281):
            sourceAdd = sourceInventory[i][j] / 83

            # adjust pollutant source around beijing
            if (i >= 96 and i <= 104 and j >= 180 and j <= 188):
                sourceAdd = sourceAdd * 0.7
            # adjust pollutant source around shanghai
            if(i >= 60 and i <= 70 and j >=  200 and j <= 210):
                sourceAdd = sourceAdd * 0.7
            # adjust pollutant source around Zhusanjiao
            if (i >=30 and i <= 35 and j >= 168 and j <= 176):
                sourceAdd = sourceAdd * 0.5
            # adjust pollutant source around Hu'nan, ChongQing
            if (i >= 40 and i <= 58 and j >= 128 and j <= 190):
                sourceAdd = sourceAdd * 0.6
            # adjust pollutant source around south Hebei
            if (i >= 76 and i <= 92 and j >= 176 and j <= 185):
                sourceAdd = sourceAdd * 0.8
            # adjust pollutant source around west Guizhou
            if (i >= 41  and i <= 64 and j >= 136 and j <= 144):
                sourceAdd = sourceAdd * 0.4
            # adjust pollutant source around zhengzhou
            if (i >= 78 and i <= 84 and j >= 172 and j <= 178):
                sourceAdd = sourceAdd * 0.4

            # 30 41 105 116
            # control pollutant source
            if (i >= 60 and i <= 104 and j >= 140 and j <= 184):
                if sourceAdd > 20:
                    sourceAdd = 20

            # adjust dongbei pollutant higher
            if (i >= 104 and i <= 120 and j >= 216 and j <= 224):
                sourceAdd = sourceAdd * 2
            # adjust Ulumuqi pollutant higher
            if (i >= 108 and i <= 124 and j >= 56 and j <= 80):
                sourceAdd = sourceAdd * 1.2

            # taiwan pollutant adjust higher.
            if (i >= 24 and i <= 42 and j >= 200 and j <= 208):
                sourceAdd = sourceAdd * 5.5

            # Korean pollutant adjust higher.
            if (i >= 78 and i <= 92 and j >= 220 and j <= 240):
                sourceAdd = sourceAdd * 2.5

            # adjust Indian pollutant higher
            if (i >= 24 and i <= 80 and j >= 24 and j <= 88):
                sourceAdd = sourceAdd * 4


            if(sourceAdd > 40):
                sourceAdd = 40

            if isInGrid(i - 1, j - 1):
                sourceContribution[i - 1][j - 1] += sourceAdd * 0.05
            if isInGrid(i - 1, j):
                sourceContribution[i - 1][j] += sourceAdd * 0.1
            if isInGrid(i - 1, j + 1):
                sourceContribution[i - 1][j + 1] += sourceAdd * 0.05

            if isInGrid(i, j - 1):
                sourceContribution[i][j - 1] += sourceAdd * 0.1
            if isInGrid(i, j):
                sourceContribution[i][j] += sourceAdd * 0.4
            if isInGrid(i, j + 1):
                sourceContribution[i][j + 1] += sourceAdd * 0.1

            if isInGrid(i + 1, j - 1):
                sourceContribution[i + 1][j - 1] += sourceAdd * 0.05
            if isInGrid(i + 1, j):
                sourceContribution[i + 1][j] += sourceAdd * 0.1
            if isInGrid(i + 1, j + 1):
                sourceContribution[i + 1][j + 1] += sourceAdd * 0.05

    return sourceContribution

# natrual motion
def natrualPart(grbs, grbs2, pm25Grid):
    afterProcess = np.zeros((161,281))
    portionDistribution = np.zeros((161, 281))
    swi = SWIindex(grbs, grbs2)
    land = grbs.select(name='Land-sea coverage (nearest neighbor) [land=1,sea=0]')[0]
    land = land.values

    for i in range(161):
        for j in range(281):
            portionDistribution[i][j] = 1 - (1 - swi[i][j] * 1.2) / 6.5 # 7.0
            if(land[i][j] == 0):
                if(portionDistribution[i][j] < 0.94):
                    portionDistribution[i][j] = 0.94
            if(j >= 220 and j <= 280 and i > 132):
                if (portionDistribution[i][j] > 0.98):
                    portionDistribution[i][j] = 0.98
            if (j <= 60 and i > 80):
                if (portionDistribution[i][j] > 0.98):
                    portionDistribution[i][j] = 0.98

    for i in range(161):
        for j in range(281):
            pm25Grid[i][j] = pm25Grid[i][j] * portionDistribution[i][j]

            if isInGrid(i - 2, j - 2):
                afterProcess[i - 2][j - 2] += pm25Grid[i][j] * outer
            if isInGrid(i - 2, j - 1):
                afterProcess[i - 2][j - 1] += pm25Grid[i][j] * subouter
            if isInGrid(i - 2, j):
                afterProcess[i - 2][j] += pm25Grid[i][j] * subcenter
            if isInGrid(i - 2, j + 1):
                afterProcess[i - 2][j + 1] += pm25Grid[i][j] * subouter
            if isInGrid(i - 2, j + 2):
                afterProcess[i - 2][j + 2] += pm25Grid[i][j] * outer

            if isInGrid(i - 1, j - 2):
                afterProcess[i - 1][j - 2] += pm25Grid[i][j] * subouter
            if isInGrid(i - 1, j - 1):
                afterProcess[i - 1][j - 1] += pm25Grid[i][j] * subcenter
            if isInGrid(i - 1, j):
                afterProcess[i - 1][j] += pm25Grid[i][j] * cocenter
            if isInGrid(i - 1, j + 1):
                afterProcess[i - 1][j + 1] += pm25Grid[i][j] * subcenter
            if isInGrid(i - 1, j + 2):
                afterProcess[i - 1][j + 2] += pm25Grid[i][j] * subouter


            if isInGrid(i, j - 2):
                afterProcess[i][j - 2] += pm25Grid[i][j] * subcenter
            if isInGrid(i, j - 1):
                afterProcess[i][j - 1] += pm25Grid[i][j] * cocenter
            if isInGrid(i, j):
                afterProcess[i][j] += pm25Grid[i][j] * center
            if isInGrid(i, j + 1):
                afterProcess[i][j + 1] += pm25Grid[i][j] * cocenter
            if isInGrid(i, j + 2):
                afterProcess[i][j + 2] += pm25Grid[i][j] * subcenter

            if isInGrid(i + 1, j - 2):
                afterProcess[i + 1][j - 2] += pm25Grid[i][j] * subouter
            if isInGrid(i + 1, j - 1):
                afterProcess[i + 1][j - 1] += pm25Grid[i][j] * subcenter
            if isInGrid(i + 1, j):
                afterProcess[i + 1][j] += pm25Grid[i][j] * cocenter
            if isInGrid(i + 1, j + 1):
                afterProcess[i + 1][j + 1] += pm25Grid[i][j] * subcenter
            if isInGrid(i + 1, j + 2):
                afterProcess[i + 1][j + 2] += pm25Grid[i][j] * subouter

            if isInGrid(i + 2, j - 2):
                afterProcess[i + 2][j - 2] += pm25Grid[i][j] * outer
            if isInGrid(i + 2, j - 1):
                afterProcess[i + 2][j - 1] += pm25Grid[i][j] * subouter
            if isInGrid(i + 2, j):
                afterProcess[i + 2][j] += pm25Grid[i][j] * subcenter
            if isInGrid(i + 2, j + 1):
                afterProcess[i + 2][j + 1] += pm25Grid[i][j] * subouter
            if isInGrid(i + 2, j + 2):
                afterProcess[i + 2][j + 2] += pm25Grid[i][j] * outer

    # grid INFO: Precipitation rate
    #  kg m**-2 s**-1 (Average)
    try:
        PR = grbs.select(name = "Precipitation rate")[0]
        PR = PR.values
        for i in range(161):
            for j in range(281):
                gridPR = PR[i][j] * 3600 # unit: mm

                groundedPortion = 1
                if gridPR > 0:
                    groundedPortion = - 0.2 * gridPR + 1  # 1: 0.9; 5: 0.5

                if groundedPortion < 0.2:
                    groundedPortion = 0.2
                elif groundedPortion > 1 and gridPR > 0:
                    groundedPortion = 1

                afterProcess[i][j] = afterProcess[i][j] * groundedPortion
    except:
        print("[Warning] No Precipitation rate INFO, Skip.")

    return [afterProcess, swi]

# return 0 ~ 1(0 ~ 100 %)
def SWIindex(grbs, grbs2):
    sourceInventoryPortion = np.zeros((161, 281))

    surfacePressure = grbs.select(name='Surface pressure')[0].values

    humidity_2m = grbs.select(name='2 metre relative humidity')[0].values
    VV = grbs.select(name='Vertical velocity')
    verticalVelocity_300 = VV[4].values
    verticalVelocity_350 = VV[5].values
    verticalVelocity_400 = VV[6].values
    verticalVelocity_450 = VV[7].values
    verticalVelocity_500 = VV[8].values
    verticalVelocity_550 = VV[9].values
    verticalVelocity_600 = VV[10].values
    verticalVelocity_650 = VV[11].values
    verticalVelocity_700 = VV[12].values
    verticalVelocity_750 = VV[13].values
    verticalVelocity_800 = VV[14].values
    verticalVelocity_850 = VV[15].values
    verticalVelocity_900 = VV[16].values
    verticalVelocity_950 = VV[18].values
    # MSLP = grbs.select(name='MSLP (Eta model reduction)')[0].values
    PBLH = grbs.select(name='Planetary boundary layer height')[0].values
    PBLH2 = grbs2.select(name='Planetary boundary layer height')[0].values
    # T_surface = grbs.select(name='Temperature')[31].values  # surface temperature
    T = grbs.select(name='Temperature')
    T_1000 = T[30].values  # 1000hpa temperature
    T_950 = T[28].values  # 950hpa temperature
    T_900 = T[26].values  # 950hpa temperature
    T_850 = T[25].values  # 850hpa temperature
    T_800 = T[24].values  # 800hpa temperature
    T_750 = T[23].values  # 750hpa temperature
    T_700 = T[22].values  # 700hpa temperature
    T_650 = T[21].values  # 650hpa temperature
    T_600 = T[20].values  # 600hpa temperature
    T_550 = T[19].values  # 550hpa temperature
    T_500 = T[18].values  # 500hpa temperature
    T_450 = T[17].values  # 450hpa temperature
    T_400 = T[16].values  # 400hpa temperature
    T_350 = T[15].values  # 350hpa temperature

    for i in range(161):
        for j in range(281):
            deltaT_subIndex = 0 # 0 ~ 6
            PBLH_subIndex = 0 # 0 ~ 4
            vertical_subIndex = 0 # 0 ~ 3
            humid_subIndex = 0 # 0 ~ 3

            # deltaT subIndex
            # vertical velocity

            # deltaT_1_2: delta T btw level 1 and 2
            # deltaT_1_3: delta T btw level 1 and 3
            # deltaT_2_3: delta T btw level 2 and 3
            surfacePressure[i][j] = surfacePressure[i][j] / 100

            deltaT_1_2 = 0
            deltaT_1_3 = 0
            deltaT_2_3 = 0
            if surfacePressure[i][j] > 1000:
                deltaT_1_2 = T_950[i][j] - T_1000[i][j]
                deltaT_2_3 = T_900[i][j] - T_950[i][j]
                deltaT_1_3 = T_900[i][j] - T_1000[i][j]
                if (verticalVelocity_950[i][j] < 0.2):
                    if (verticalVelocity_850[i][j] < 0.2):
                        vertical_subIndex = 4
                    else:
                        vertical_subIndex = 3
            elif surfacePressure[i][j] > 950:
                deltaT_1_2 = T_900[i][j] - T_950[i][j]
                deltaT_2_3 = T_850[i][j] - T_900[i][j]
                deltaT_1_3 = T_850[i][j] - T_950[i][j]

                if (verticalVelocity_900[i][j] < 0.2):
                    if (verticalVelocity_800[i][j] < 0.2):
                        vertical_subIndex = 4
                    else:
                        vertical_subIndex = 3
            elif surfacePressure[i][j] > 900:
                deltaT_1_2 = T_850[i][j] - T_900[i][j]
                deltaT_2_3 = T_800[i][j] - T_850[i][j]
                deltaT_1_3 = T_800[i][j] - T_900[i][j]

                if (verticalVelocity_850[i][j] < 0.2):
                    if (verticalVelocity_750[i][j] < 0.2):
                        vertical_subIndex = 4
                    else:
                        vertical_subIndex = 3
            elif surfacePressure[i][j] > 850:
                deltaT_1_2 = T_800[i][j] - T_850[i][j]
                deltaT_2_3 = T_750[i][j] - T_800[i][j]
                deltaT_1_3 = T_750[i][j] - T_850[i][j]

                if (verticalVelocity_800[i][j] < 0.2):
                    if (verticalVelocity_700[i][j] < 0.2):
                        vertical_subIndex = 4
                    else:
                        vertical_subIndex = 3
            elif surfacePressure[i][j] > 800:
                deltaT_1_2 = T_750[i][j] - T_800[i][j]
                deltaT_2_3 = T_700[i][j] - T_750[i][j]
                deltaT_1_3 = T_700[i][j] - T_800[i][j]

                if (verticalVelocity_750[i][j] < 0.2):
                    if (verticalVelocity_650[i][j] < 0.2):
                        vertical_subIndex = 4
                    else:
                        vertical_subIndex = 3
            elif surfacePressure[i][j] > 750:
                deltaT_1_2 = T_700[i][j] - T_750[i][j]
                deltaT_2_3 = T_650[i][j] - T_700[i][j]
                deltaT_1_3 = T_650[i][j] - T_750[i][j]

                if (verticalVelocity_700[i][j] < 0.2):
                    if (verticalVelocity_650[i][j] < 0.2):
                        vertical_subIndex = 4
                    else:
                        vertical_subIndex = 3
            elif surfacePressure[i][j] > 700:
                deltaT_1_2 = T_650[i][j] - T_700[i][j]
                deltaT_2_3 = T_600[i][j] - T_650[i][j]
                deltaT_1_3 = T_600[i][j] - T_700[i][j]

                if (verticalVelocity_650[i][j] < 0.2):
                    if (verticalVelocity_600[i][j] < 0.2):
                        vertical_subIndex = 4
                    else:
                        vertical_subIndex = 3
            elif surfacePressure[i][j] > 650:
                deltaT_1_2 = T_600[i][j] - T_650[i][j]
                deltaT_2_3 = T_550[i][j] - T_600[i][j]
                deltaT_1_3 = T_550[i][j] - T_650[i][j]

                if (verticalVelocity_600[i][j] < 0.2):
                    if (verticalVelocity_550[i][j] < 0.2):
                        vertical_subIndex = 4
                    else:
                        vertical_subIndex = 3
            elif surfacePressure[i][j] > 600:
                deltaT_1_2 = T_550[i][j] - T_600[i][j]
                deltaT_2_3 = T_500[i][j] - T_550[i][j]
                deltaT_1_3 = T_500[i][j] - T_600[i][j]

                if (verticalVelocity_550[i][j] < 0.2):
                    if (verticalVelocity_500[i][j] < 0.2):
                        vertical_subIndex = 4
                    else:
                        vertical_subIndex = 3
            elif surfacePressure[i][j] > 550:
                deltaT_1_2 = T_500[i][j] - T_550[i][j]
                deltaT_2_3 = T_450[i][j] - T_500[i][j]
                deltaT_1_3 = T_450[i][j] - T_550[i][j]

                if (verticalVelocity_500[i][j] < 0.2):
                    if (verticalVelocity_450[i][j] < 0.2):
                        vertical_subIndex = 4
                    else:
                        vertical_subIndex = 3
            elif surfacePressure[i][j] > 500:
                deltaT_1_2 = T_450[i][j] - T_500[i][j]
                deltaT_2_3 = T_400[i][j] - T_450[i][j]
                deltaT_1_3 = T_400[i][j] - T_500[i][j]

                if (verticalVelocity_450[i][j] < 0.2):
                    if (verticalVelocity_400[i][j] < 0.2):
                        vertical_subIndex = 4
                    else:
                        vertical_subIndex = 3
            elif surfacePressure[i][j] > 450:
                deltaT_1_2 = T_400[i][j] - T_450[i][j]
                deltaT_2_3 = T_350[i][j] - T_400[i][j]
                deltaT_1_3 = T_350[i][j] - T_450[i][j]

                if (verticalVelocity_350[i][j] < 0.2):
                    if (verticalVelocity_300[i][j] < 0.2):
                        vertical_subIndex = 4
                    else:
                        vertical_subIndex = 3

            deltaT_subIndex = (deltaT_1_2 + 1.5) + (deltaT_2_3 + 1.5)

            # PBLH subIndex
            if(PBLH[i][j] < 1500):
                PBLH_subIndex = 3 / 15 * (15 - PBLH[i][j] / 100.0)
                if(PBLH_subIndex > 4):
                    PBLH_subIndex = 4

            PBLHchange = -(PBLH2[i][j] - PBLH[i][j]) / PBLH2[i][j] # negative means bad
            if(PBLHchange > 2):
                PBLHchange = 2
            if(PBLHchange < -2):
                PBLHchange = -2

            # humid
            humid_subIndex = (humidity_2m[i][j] - 70) / 10.0
            if(humid_subIndex < 0):
                humid_subIndex = 0

            # sum up all subIndex
            sourceInventoryPortion[i][j] = deltaT_subIndex + PBLH_subIndex \
                                           + vertical_subIndex + humid_subIndex + PBLHchange
            sourceInventoryPortion[i][j] = sourceInventoryPortion[i][j] / 11.5

            if(surfacePressure[i][j] < 950):
                sourceInventoryPortion[i][j] = sourceInventoryPortion[i][j] * (1 - 0.002 * (950 - surfacePressure[i][j]))

            if(sourceInventoryPortion[i][j] < 0.6):
                sourceInventoryPortion[i][j] = 0.6
            elif(sourceInventoryPortion[i][j] > 1):
                sourceInventoryPortion[i][j] = 1

    return sourceInventoryPortion

def isInGrid(row, col):
    if(row >= 0 and row < 161 and col >= 0 and col < 281):
        return True
    else:
        return False

######################### ANALYSIS HELPER FUNCTIONS END ###########################

######################### OUTPUT HELPER FUNCTIONS BEGIN ###########################
def resultout2hd5(gridArray, time, fct, GFSheadHour):
    lon = []
    lat = []
    for i in range(0, 161):
        lat.append(15 + i * 0.25)
    for i in range(0, 281):
        lon.append(70 + i * 0.25)
    x, y = np.meshgrid(lon, lat)


    f = h5.File('output/' + GFSheadHour + '/L126_H13_' + fct + '.hdf5', 'w')
    f.create_dataset('time', data=time)
    f.create_dataset('forecast hour', data=fct)
    f.create_dataset('lon', data=x)
    f.create_dataset('lat', data=y)
    f.create_dataset('pm25', data=gridArray)
    f.close()

def swiout2hd5(gridArray, time, fct, GFSheadHour):
    lon = []
    lat = []
    for i in range(0, 161):
        lat.append(15 + i * 0.25)
    for i in range(0, 281):
        lon.append(70 + i * 0.25)
    x, y = np.meshgrid(lon, lat)


    f = h5.File('output/' + GFSheadHour + '/SWI_' + fct + '.hdf5', 'w')
    f.create_dataset('time', data=time)
    f.create_dataset('forecast hour', data=fct)
    f.create_dataset('lon', data=x)
    f.create_dataset('lat', data=y)
    f.create_dataset('SWI', data=gridArray)
    f.close()

def sourceout2hd5(gridArray, time, fct, GFSheadHour):
    lon = []
    lat = []
    for i in range(0, 161):
        lat.append(15 + i * 0.25)
    for i in range(0, 281):
        lon.append(70 + i * 0.25)
        x, y = np.meshgrid(lon, lat)
    f = h5.File('output/' + GFSheadHour + '/source_' + fct + '.hdf5', 'w')
    f.create_dataset('time', data=time)
    f.create_dataset('forecast hour', data=fct)
    f.create_dataset('lon', data=x)
    f.create_dataset('lat', data=y)
    f.create_dataset('source', data=gridArray)
    f.close()

def wind2hd5(gridArray, time, fct, GFSheadHour):
    lon = []
    lat = []
    for i in range(0, 161):
        lat.append(15 + i * 0.25)
    for i in range(0, 281):
        lon.append(70 + i * 0.25)
    x, y = np.meshgrid(lon, lat)
    f = h5.File('output/' + GFSheadHour + '/wind' + fct + '.hdf5', 'w')
    f.create_dataset('time', data=time)
    f.create_dataset('forecast hour', data=fct)
    f.create_dataset('lon', data=x)
    f.create_dataset('lat', data=y)
    f.create_dataset('wind', data=gridArray)
    f.close()
######################### OUTPUT HELPER FUNCTIONS END ###########################

def plot(gridArray, windu, windv, title, GFSheadHour):
    lon = []
    lat = []
    for i in range(0, 161):
        lat.append(15 + i * 0.25)
    for i in range(0, 281):
        lon.append(70 + i * 0.25)
    x, y = np.meshgrid(lon, lat)

    m = Basemap(llcrnrlon=70, llcrnrlat=15, urcrnrlon=140, urcrnrlat=55, resolution='l', area_thresh=100)
    x, y = m(x, y)
    fig = plt.figure(figsize=(10, 7), dpi=200)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')

    my_cmap = mpl.colors.LinearSegmentedColormap('my_colormap', color.pm25_better, 256)
    norm = mpl.colors.Normalize(0, 500)
    m.contourf(x, y, gridArray, 500, cmap=my_cmap, norm=norm)  # ,norm=norm

    skip = slice(None, None, 5)
    m.barbs(x[skip, skip], y[skip, skip], windu[skip, skip], windv[skip, skip], length=3.5,
            sizes=dict(emptybarb=0, spacing=0.2, height=0.5), barb_increments=dict(half=2, full=4, flag=20),
            linewidth=0.2, color='black')

    plt.title('Concentration of PM2.5(East Asia) Forecast - ' + title + '\nL126_H13 Regional Air Quality Model [Experimental beta 1.0.0] @Louis-He',
              loc='left', fontsize=11)
    m.drawparallels(np.arange(0, 65, 10), labels=[1, 0, 0, 0], fontsize=8, linewidth=0.5, color='dimgrey',
                    dashes=[1, 1])
    m.drawmeridians(np.arange(65., 180., 10), labels=[0, 0, 0, 1], fontsize=8, linewidth=0.5, color='dimgrey',
                    dashes=[1, 1])
    m.drawcoastlines(linewidth=0.5)
    m.drawstates(linewidth=0.4, color='dimgrey')
    m.readshapefile('cnhimap', 'states', drawbounds=True, linewidth=0.5, color='black')
    ax2 = fig.add_axes([0.92, 0.16, 0.018, 0.67])
    # clevs1 = [0, 35, 75, 115, 150, 250, 500]
    # cmap1 = mpl.colors.ListedColormap([[0, 228 / 255, 0], [1, 1, 0],
    #                                    [1, 126 / 255, 0], [1, 0, 0],
    #                                    [153 / 255, 0, 76 / 255], [126 / 255, 0, 35 / 255]])
    clevs1 = [0, 15, 35, 50, 75, 100, 115, 130, 150, 200, 250, 500]
    cmap1 = mpl.colors.ListedColormap([[0, 255 / 255, 0], [0, 228 / 255, 0], [178 / 255, 1, 0], [1, 1, 0],
                                       [1, 196 / 255, 0], [1, 126 / 255, 0], [1, 70 / 255, 0], [1, 0, 0],
                                       [204 / 255, 0, 128 / 255], [153 / 255, 0, 76 / 255], [126 / 255, 0, 35 / 255]])
    norm1 = mpl.colors.BoundaryNorm(clevs1, cmap1.N)
    cbar1 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap1, spacing='uniform', norm=norm1, ticks=clevs1,
                                      orientation='vertical', drawedges=False)
    cbar1.ax.set_ylabel('pm2.5(ug/m**3)', size=8)
    cbar1.ax.tick_params(labelsize=8)
    plt.savefig('output/' + GFSheadHour + '/' + title + '.png', bbox_inches='tight')
    fig = fig.clear()
    plt.clf()
    plt.close()

def plotSWI(gridArray, title, GFSheadHour):
    lon = []
    lat = []
    for i in range(0, 161):
        lat.append(15 + i * 0.25)
    for i in range(0, 281):
        lon.append(70 + i * 0.25)
    x, y = np.meshgrid(lon, lat)

    m = Basemap(llcrnrlon=70, llcrnrlat=15, urcrnrlon=140, urcrnrlat=55, resolution='l', area_thresh=100)
    x, y = m(x, y)
    fig = plt.figure(figsize=(10, 7), dpi=200)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')

    my_cmap = mpl.colors.LinearSegmentedColormap('my_colormap', color.pm25, 256)
    norm = mpl.colors.Normalize(0, 500)
    m.contourf(x, y, gridArray, 500, cmap=my_cmap, norm=norm)  # ,norm=norm

    plt.title('Concentration of PM2.5(China) Forecast - ' + title + '\nL126_H13 Regional Air Quality Model [In trial] @Louis-He',
              loc='left', fontsize=11)
    m.drawparallels(np.arange(0, 65, 10), labels=[1, 0, 0, 0], fontsize=8, linewidth=0.5, color='dimgrey',
                    dashes=[1, 1])
    m.drawmeridians(np.arange(65., 180., 10), labels=[0, 0, 0, 1], fontsize=8, linewidth=0.5, color='dimgrey',
                    dashes=[1, 1])
    m.drawcoastlines(linewidth=0.5)
    m.drawstates(linewidth=0.4, color='dimgrey')
    m.readshapefile('cnhimap', 'states', drawbounds=True, linewidth=0.5, color='black')
    ax2 = fig.add_axes([0.92, 0.16, 0.018, 0.67])
    clevs1 = [0, 35, 75, 115, 150, 250, 500]
    cmap1 = mpl.colors.ListedColormap([[0, 228 / 255, 0], [1, 1, 0],
                                       [1, 126 / 255, 0], [1, 0, 0],
                                       [153 / 255, 0, 76 / 255], [126 / 255, 0, 35 / 255]])
    norm1 = mpl.colors.BoundaryNorm(clevs1, cmap1.N)
    cbar1 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap1, spacing='uniform', norm=norm1, ticks=clevs1,
                                      orientation='vertical', drawedges=False)
    cbar1.ax.set_ylabel('pm2.5(ug/m**3)', size=8)
    cbar1.ax.tick_params(labelsize=8)
    plt.savefig('output/' + GFSheadHour + '/' + title + '.png', bbox_inches='tight')
    fig = fig.clear()
    plt.clf()
    plt.close()

def mainOperation(GFSheadHour, hour, gfsHour, gfsHourNext, GFSBeforeHeadStr):
    ######################### INPUT DATASET BEGIN ###########################
    # 70 ~ 140: 70
    # 15 ~ 55: 15
    # shape: 161 * 281
    start = times.time()

    if len(gfsHour) == 3:
        grbs = pygrib.open('rawfile/' + GFSheadHour + '.f' + gfsHour)
    else:
        grbs = pygrib.open('rawfile/' + GFSheadHour + '.f0' + gfsHour)

    if len(gfsHourNext) == 3:
        grbs2 = pygrib.open('rawfile/' + GFSheadHour + '.f' + gfsHourNext)
    else:
        grbs2 = pygrib.open('rawfile/' + GFSheadHour + '.f0' + gfsHourNext)

    # 70 ~ 140: 70
    # 15 ~ 55: 15
    # shape: 161 * 281
    pm2_5InitialInfo = readInitialPM2_5Field(hour, GFSheadHour)
    pm2_5InitialField = pm2_5InitialInfo['pm25']
    time = pm2_5InitialInfo['time']

    timeStr = str(np.array(time))

    timeNum = int(timeStr[timeStr.find('T') + 1 : timeStr.find(':00:00')])
    fctNum = timeNum + int(hour)

    while(fctNum > 24):
        fctNum = fctNum - 24

    pm2_5Inventory = readinventoryPM2_5(month, fctNum)

    ######################### INPUT DATASET END ###########################

    ################# OUTPUT DATASET DECLEARATION BEGIN ###################

    pm2_5Forecast = np.zeros((161, 281))

    ################# OUTPUT DATASET DECLEARATION END #####################

    #### SIMULATION START ####
    fct = int(hour) + 1
    if fct < 10:
        fct = '0' + str(fct)
    else:
        fct = str(fct)

    if fct == '01':
        pm2_5InitialField = preProcess(pm2_5InitialField, grbs, pm2_5Inventory, GFSBeforeHeadStr)
    wind = windComponent(pm2_5InitialField, grbs, grbs2)
    windPart = wind['windComponent']
    newPollutant = newPollutantFromSource(grbs, pm2_5Inventory)

    pm2_5Forecast = windPart + newPollutant
    natrual = natrualPart(grbs, grbs2, pm2_5Forecast)
    pm2_5Forecast = natrual[0]

    if fct == '01':
        plot(pm2_5InitialField, wind['windu'], wind['windv'], time + '+00hr', GFSheadHour)

    # sourceout2hd5(newPollutant, time, fct)
    # wind2hd5(windPart, time, fct)
    resultout2hd5(pm2_5Forecast, time, fct, GFSheadHour)
    swiout2hd5(natrual[1], time, fct, GFSheadHour)
    plot(pm2_5Forecast, wind['windu'], wind['windv'], time + '+' + fct + 'hr', GFSheadHour)
    print('[SUCCESS] calculate ' + fct + ' hour Forecast')
    print('total: ', times.time() - start, ' s')

    #### SIMULATION END ####

def checkingFileIntegrity():
    print('[Wait] Checking File Integrity.')
    files = os.listdir('rawfile')
    for file in files:
        # print(file)
        if file[-4:] == 'f084':
            print('[Success] File Integrity check.')
            return True
    return False

# return timeStamp BJY
def GFSInitTime():
    files = os.listdir('rawfile')
    for file in files:
        if file[-4:] == 'f000':
            time = file[file.find('gfs.GFS') + len('gfs.GFS') : file.find('.f000')]
            datestr = times.strptime(time, "%Y%m%d%H")
            timeStamp = times.mktime(datestr) + 8 * 60 * 60
            return [timeStamp, timeStamp - 14 * 60 * 60] # return latest and one time slot beforehead

# return string of GFS hour
def GFSInitTimeStr():
    files = os.listdir('rawfile')
    for file in files:
        if file[-4:] == 'f000':
            time = file[: file.find('.f000')]
            return time

# return timeStamp BJY
def airInitTime():
    file = h5.File('air.hdf5', 'r')
    time = file['time']
    time = str(np.array(time))
    datestr = times.strptime(time, "%Y-%m-%dT%H:%M:%S")
    timeStamp = times.mktime(datestr)
    return timeStamp

def determineOffset():
    airTime = airInitTime()
    gfsTime = GFSInitTime()[0]

    print (airTime - gfsTime)
    return int((airTime - gfsTime) / 3600)

while(not checkingFileIntegrity()):
    print('[WARNING]Wait for GFS files to download')
    times.sleep(60)

print('[Wait] Determine GFS hour offset')

offset = determineOffset()
GFSHead = GFSInitTimeStr()
GFSBeforeHead = GFSInitTime()[1] #  UTC + 0
GFSBeforeHeadStr = "gfs.GFS" + times.strftime('%Y%m%d%H', times.localtime(GFSBeforeHead))

os.system('rm -rf output/' + GFSHead)
os.system('mkdir output/' + GFSHead)
print('[INFO] Created output directory: output/' + GFSHead)

for i in range(96):
    gfsHour = i + offset
    gfsHourNext = gfsHour + 1
    if i < 10:
        i = '0' + str(i)
    else:
        i = str(i)

    if gfsHour < 10:
        gfsHour = '0' + str(gfsHour)
    else:
        gfsHour = str(gfsHour)

    if gfsHourNext < 10:
        gfsHourNext = '0' + str(gfsHourNext)
    else:
        gfsHourNext = str(gfsHourNext)

    mainOperation(GFSHead, i, gfsHour, gfsHourNext, GFSBeforeHeadStr)
os.system('python3 point_Forecast.py --GFSheadHour ' + str(GFSHead))