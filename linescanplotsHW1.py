
# coding: utf-8

# # EECS192 Spring 2017 Track Finding from 1D line sensor data

# In[6]:

# changed to use 8 bit compressed line sensor values
# data format: 128 comma separated values, last value in line has space, not comma
# line samples are about 10 ms apart
#  csv file format time in ms, 128 byte array, velocity
# note AGC has already been applied to data, and camera has been calibrated for illumination effects


# In[8]:

import numpy as np
import scipy.ndimage as ndi
from scipy.signal import gaussian
from scipy.optimize import curve_fit
import scipy as sp
import matplotlib.pyplot as plt
import sys

import scipy.ndimage as ndi  # useful for 1d filtering functions

plt.close("all")   # try to close all open figs

# In[9]:

# Graphing helper function
def setup_graph(title='', x_label='', y_label='', fig_size=None):
    fig = plt.figure()
    if fig_size != None:
        fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


# Line scan plotting function.
# 

# In[10]:

def plot_frame(linearray):
    nframes = np.size(linearray)/128
    n = range(0,128)
    print 'number of frames', nframes
    print 'size of line', np.size(linearray[0,:])
    for i in range(0, nframes-1):
        setup_graph(title='$x[n]$', x_label='$n$', y_label='row'+str(i)+' $ xa[n]$', fig_size=(15,2))
        plt.subplot(1,3,1)
        _ = plt.plot(n,linearray[0,:])
        plt.subplot(1,3,2)
        _ = plt.plot(n,linearray[i,:])
    # plot simple difference between frame i and first frame
        plt.subplot(1,3,3)
        _ = plt.plot(n,linearray[i,:] - linearray[0,:])
        plt.ylabel('Frame n - Frame 0')


# ### grayscale plotting of line function:
# 

# In[11]:

CAMERA_LENGTH = 128
INTENSITY_MIN = 0
INTENSITY_MAX = 255
def plot_gray(fig, camera_data):
  # x fencepost positions of each data matrix element
  x_mesh = []
  for i in range(0, len(camera_data)+1):
    x_mesh.append([i-0.5] * (CAMERA_LENGTH + 1))
  x_mesh = np.array(x_mesh)
  
  # y fencepost positions of each data matrix element
  y_array = range(0, CAMERA_LENGTH + 1)
  y_array = list(map(lambda x: x - 0.5, y_array))
  y_mesh = np.array([y_array] * (len(camera_data)+1))
    
  data_mesh = np.array(camera_data)
  vmax1 = np.max(data_mesh)
  data_mesh = INTENSITY_MAX * data_mesh/vmax1  # normalize intensity
  
  fig.set_xlim([-0.5, len(camera_data) - 0.5])
  fig.set_ylim([-8.5, CAMERA_LENGTH - 0.5])

  fig.pcolorfast(x_mesh, y_mesh, data_mesh,
      cmap='gray', vmin=INTENSITY_MIN, vmax=INTENSITY_MAX,
      interpolation='None')


  ## plot three pictures

def plot_all(filename):

  
    ############# plots ###########
    # fig=plt.figure()
    fig = plt.figure(figsize = (16, 3))
    fig.set_size_inches(13, 4)
    fig.suptitle("%s\n" % (filename))     
    ax = plt.subplot(1, 1, 1)
    # plot_gray(ax, linescans[0:1000])  # plot smaller range if hard too see
    plot_gray(ax, linescans) 



    ############# plot of velocities
    # fig = plt.figure(figsize = (8, 4))
    # fig.set_size_inches(13, 4)
    # fig.suptitle("velocities %s\n" % (filename))   
    # plt.xlabel('time [ms]')
    # plt.ylabel('velocity (m/s)')  
    # plt.plot(times,velocities)

    ###############plot of found track position 

    fig = plt.figure(figsize = (8, 4))
    fig.set_size_inches(13, 4)
    fig.suptitle("track center %s\n" % (filename))   
    plt.xlabel('time [ms]')
    plt.ylabel('track center')  
    plt.plot(times,track_center_list)
    plt.show()




def findTrackCenter(linescans,track_center_list):
  length = len(linescans)
  for i in range(0,length):
    maxValueIndex = np.argmax(linescans[i])
    track_center_list[i] = maxValueIndex
  return track_center_list


def isContainTrack(line,track_center_list,track_found_list):
  length = len(line)



# In[12]:

### inputs:
# linescans - An array of length n where each element is an array of length 128. Represents n frames of linescan data.

### outputs:
# track_center_list - A length n array of integers from 0 to 127. Represents the predicted center of the line in each frame.
# track_found_list - A length n array of booleans. Represents whether or not each frame contains a detected line.
# cross_found_list - A length n array of booleans. Represents whether or not each frame contains a crossing.

def find_track(linescans):
    n = len(linescans)
    track_center_list = n * [64]
    track_found_list = n * [True]
    cross_found_list = n * [False]

    plotarray = np.arange(0,128,1);

    ### Code to be added here
    # track_center_list = findTrackCenter(linescans,track_center_list)
    ###
    # track_found_list = isContainTrack(linescans,track_center_list,track_found_list)
    ###
    for j in range(len(linescans)):
    # for j in range(380,381):
        curr = linescans[j]


        #remove loss from start
        for curri in range(0,7): 
            curr[curri] = curr[curri] * 1.45;

        # curr = ndi.gaussian_filter1d(curr,7)
        
        #curr[50:75] = curr[25:50]
        #normalize
        curr = curr - min(curr)
        curr = curr / max(curr) * 255

        #make a base gaussian for center
        #center at arounrd 62
        #from 40 to 85
        c_gauss = gaussian(128,5) * 255

        #get starter 'best' values
        bestCDiff = sys.maxint
        bestCGauss = c_gauss
        bestCIndex = 0

        #find center
        for i in range(128):
            shifted = np.concatenate((c_gauss[i:128], c_gauss[0:i]), axis=0)
            diff = curr - shifted
            ssd = sum(diff**2)
            if ssd<bestCDiff:
                bestCDiff = ssd
                bestCGauss = shifted
                bestCIndex = 128-i+64
                if (bestCIndex>128):
                    bestCIndex = bestCIndex - 128
        
        #check if the gaussian missed the track
        #  by looking at the difference

        if ( (max(linescans[j]) - np.mean(linescans[j])) < 20):
          track_found_list[j] = False
        else:
          track_found_list[j] = bestCDiff < 900000
        
        track_center_list[j] = bestCIndex


    
        #box to detect cross
        box_width = 40
        x_box = np.zeros(128)
        x_box[64-box_width/2:64 + box_width/2] = np.ones(box_width) * 140
        
        #get starter 'best' values for cross
        bestXDiff = sys.maxint
        bestXBox = x_box
        bestXIndex = 0

        #find cross
        for i in range(128):
            shifted = np.concatenate((x_box[i:128], x_box[0:i]), axis=0)
            diff = curr - shifted
            ssd = sum(diff**2)
            if ssd<bestXDiff:
                bestXDiff = ssd
                bestXBox = shifted
                bestXIndex = 128-i+64
                if (bestXIndex>128):
                    bestXIndex = bestXIndex - 128
        

        if ( np.var(linescans[j] - linescans[j - 1]) < 2.4):
          cross_found_list[j] = True
        else:
          cross_found_list[j] = abs(bestXIndex - bestCIndex) > box_width/2


    return track_center_list, track_found_list, cross_found_list






################
# need to use some different tricks to read csv file
import csv
filename = 'natcar2016_team1.csv'
#filename = 'natcar2016_team1_short.csv'
csvfile=open(filename, 'rb')
telemreader=csv.reader(csvfile, delimiter=',', quotechar='"')
# Actual Spring 2016 Natcar track recording by Team 1.
telemreader.next() # discard first line
telemdata = telemreader.next() # format time in ms, 128 byte array, velocity
linescans=[]  # linescan array
times=[]
velocities=[]
for row in telemreader:
    times.append(eval(row[0])) # sample time
    velocities.append(eval(row[2])) # measured velocity
    line = row[1] # get scan data
    arrayline=np.array(eval(line)) # convert line to an np array
    linescans.append(arrayline)
# print 'scan line0:', linescans[0]
# print 'scan line1:', linescans[1]


track_center_list, track_found_list, cross_found_list = find_track(linescans)

for i, (track_center, track_found, cross_found) in enumerate(zip(track_center_list, track_found_list, cross_found_list)):
   print 'scan # %d center at %d. Track_found = %s, Cross_found = %s' %(i,track_center,track_found, cross_found)

plot_all(filename)



