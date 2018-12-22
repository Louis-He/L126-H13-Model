import time
import urllib.request
from urllib.request import urlretrieve
import h5py as h5
import numpy as np
import os
from apscheduler.schedulers.blocking import BlockingScheduler

utc = 0
#decide the target init forecast time
def targetinittime():
    global downloadhour
    timechange = -1 * utc
    nowtime = time.time() + timechange * 60 * 60  # struct time of UTC
    result = time.strftime("%H", time.localtime(nowtime))  # hour of UTC time
    if int(result) >= 21:
        hourresult = '18'
    elif int(result) < 9:
        hourresult = '00'
    elif int(result) < 15:
        hourresult = '06'
    elif int(result) < 21:
        hourresult = '12'

    returnlist = []
    isdownload = []
    for i in range(0, len(downloadhour)):
        returnlist.append(hourresult)
        isdownload.append(False)

    return returnlist, isdownload

# return the URL of the latest GFS model (3 hours after the initial)
# return [0]:the URL of target file;    [1]:the filename
def decideURL2(forecasthour, isexist):
    global utc
    hourresult = ''

    timechange = -1 * utc
    nowtime = time.time() + timechange * 60 * 60 # struct time of UTC
    result = time.strftime("%H", time.localtime(nowtime))  # hour of UTC time
    #print(result)

    if int(result) <= 2:
        nowtime = time.time() + timechange * 60 * 60 - 24 * 60 * 60 # struct time of UTC, 24hours before
        hourresult = '18'
    elif int(result) <= 8:
        hourresult = '00'
    elif int(result) <= 14:
        hourresult = '06'
    elif int(result) <= 20:
        hourresult = '12'
    else:
        hourresult = '18'

    assumetime = time.strftime("%Y%m%d", time.localtime(nowtime)) + hourresult + '0000'  # hour of UTC time
    assumetimestamp = time.mktime(time.strptime(assumetime, "%Y%m%d%H%M%S"))

    if not isexist:
        assumetimestamp = assumetimestamp - 6 * 60 * 60

    #http://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t12z.pgrb2.0p25.f000&all_var=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.2017102712
    #http://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t00z.pgrb2.0p25.f000&all_var=on&subregion=&leftlon=70&rightlon=140&toplat=55&bottomlat=15&dir=%2Fgfs.2018112600

    result = time.strftime("%Y%m%d%H", time.localtime(assumetimestamp)) # final hour of UTC time

    filename = 'GFS' + result + '.f' + forecasthour
    #print(filename)
    result = 'http://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t'+ hourresult +'z.pgrb2.0p25.f' + forecasthour + '&all_var=on&subregion=&leftlon=70&rightlon=140&toplat=55&bottomlat=15&dir=%2Fgfs.' + result
    return [result, 'gfs.' + filename]

def downloadfile(forecasthour,isexist):
    #global bar
    downloadinfo = decideURL2(forecasthour,isexist)
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time() + utc * 60 * 60)) + ']Dowanloading file... from URL: ' +downloadinfo[0])
    path = 'rawfile/' + downloadinfo[1]
    try:
        urlretrieve(downloadinfo[0], path)
        print('[' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + utc * 60 * 60)) + ']Dowanloading file... from URL: ' + downloadinfo[1] + 'SUCCESS')
        return True
    except:
        if isexist:
            print('(FILE DO NOT EXIST)[' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + utc * 60 * 60)) + ']Dowanloading file... from URL: ' + downloadinfo[1] + 'ERROR')
        else:
            print('(UNEXPECTED ERR)[' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + utc * 60 * 60)) + ']Dowanloading file... from URL: ' + downloadinfo[1] + 'ERROR')
        return False

def istruelist(list):
    result = True
    for i in list:
        if not i:
            result = False

    return result

# main method
def mainmethod():
    #global bar
    global successdownload

    global downloadhour
    inittime = time.time() + (-1) * utc * 60 * 60

    #initialize the program
    initialize()

    # set the forecast hour of the file from GFS
    downloadhour = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012',
                    '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024',
                    '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
                    '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048',
                    '049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060',
                    '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', '071', '072',
                    '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '084',
                    '085', '086', '087', '088', '089', '090', '091', '092', '093', '094', '095', '096',
                    '097', '098', '099', '100', '101', '102', '103', '104', '105', '106', '107', '108',
                    '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120']
    successdownload = 0

    initialprocess = targetinittime()
    isdownload = initialprocess[1] # initial all false, true: isdownload; false: not download yet

    while not istruelist(isdownload):
    #reqiured download file
        print('[' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(
                time.time() + utc * 60 * 60)) + ']Start downloading cycle.')
        count = 0
        tmpbool = True
        for i in downloadhour:
            if not isdownload[count] and tmpbool:
                tmpbool = downloadfile(downloadhour[count], True)
                if tmpbool:
                    isdownload[count] = True
                    successdownload += 1
            count += 1
        # print(isdownload)
        print('[' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(
                time.time() + utc * 60 * 60)) + ']No more new file. Start sleeping cycle...[5 min]')
        time.sleep(120) # sleep 2 mins for another try

    finishtime = time.time() + (-1) * utc * 60 * 60
    print('*--------------------------------------------*')
    print('[' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(
                time.time() + utc * 60 * 60)) + ']Download cycle ends.')
    print('TOTAL DOWNLOADING TIME:' + str(finishtime-inittime) + 's')
    print('[INFO] Start Air Quality Forecasting...')
    os.system('python3 mainModel.py')

# decide First start time
def startmain():
    nowtime = time.time()
    result = int(time.strftime("%H%M", time.localtime(nowtime)))  # hour of UTC time
    print('nowtime: (HHMM)' + str(result))
    '''
    forecast1 = 325
    forecast2 = 925
    forecast3 = 1525
    forecast4 = 2125
    '''

    if result > 340 and result < 345:
        return True
    elif result > 940 and result < 945:
        return True
    elif result > 1540 and result < 1545:
        return True
    elif result > 2140 and result < 2145:
        return True
    return False

# run at the beginning of the program
def initialize():
    os.system('rm -rf rawfile/')
    print('['+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + utc * 60 * 60))+']'+'Erase expired rawfile')
    os.system('mkdir rawfile')
    print('[' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + utc * 60 * 60)) + ']' + 'Create rawfile folder')
    print('[' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + utc * 60 * 60)) + ']' + 'Start downloading file...')

initialize()
# mainmethod()
print('[Please Wait]System First Start: Wait for next closest GFS files download time window...')
print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))
while not startmain():
    time.sleep(60)
# main operation
scheduler = BlockingScheduler()
scheduler.add_job(mainmethod, 'interval', seconds = 6 * 60 * 60)
print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))
mainmethod()
try:
    scheduler.start()
except (KeyboardInterrupt, SystemExit):
    pass