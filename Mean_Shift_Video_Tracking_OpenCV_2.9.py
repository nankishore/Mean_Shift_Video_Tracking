# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 13:37:34 2016
@author: nandakishore
"""

import cv2
import numpy as np
import matplotlib.pyplot as mpl
import matplotlib.cm as cm
import os
import shutil
import natsort
import pylab as py
from scipy import ndimage, signal
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans, vq
from sklearn.utils import shuffle
from matplotlib.widgets import  RectangleSelector

#Global variables for initial target location
x_upper, y_upper = 0, 0
x_lower, y_lower = 0, 0

def findDerivative(image,
        axis):
    if axis == 'x':
        mask = [
            [-1, 1],
            [-1, 1]
            ]
    else:
        mask = [
            [-1, -1],
            [1, 1]
            ]
    convolved = signal.convolve2d(image,
        mask,
        mode='same'
        )
    return convolved

def kernels(
        filterHeight, filterWidth, radius,
        kernel):
    k = np.zeros((
            filterHeight,
            filterWidth))
    if kernel == 'Gaussian':
        sigH = ((filterHeight*radius)/float(2))/3
        sigW = ((filterWidth*radius)/float(2))/3
        for rows in range(0, filterHeight):
            for cols in range(0, filterWidth):
                k[rows][cols] = np.exp(-0.5*((rows
                                              - (0.5*filterHeight))**2/(sigH**2)
                                              + (cols - (0.5*filterWidth))**2/(sigW**2)))

    elif kernel == 'Uniform':
        for rows in range(0, filterHeight):
            for cols in range(0, filterWidth):
                term = ((((2*rows)/(filterHeight - 1))/radius)**2
                           + (((2*cols)/(filterWidth - 1))/radius)**2)
                if term <= 1:
                    k[rows][cols] = 1

    else:
        for rows in range(0, filterHeight):
            for cols in range(0, filterWidth):
                k[rows][cols] = (1
                                 - (2*float(rows)/(radius*filterHeight) - (1/float(radius)))**2
                                 - (2*float(cols)/(radius*filterWidth) - (1/float(radius)))**2)
                if k[rows][cols] < 0:
                    k[rows][cols] = 0

    kx = findDerivative(k,
        axis = 'x')
    ky = findDerivative(k,
        axis = 'y')
    mpl.figure(3)
    mpl.imshow(kx,
        cmap = cm.Greys_r)
    mpl.suptitle('Derivative of the kernel in X direction')
    mpl.show()
    mpl.figure(4)
    mpl.imshow(ky,
        cmap = cm.Greys_r)
    mpl.suptitle('Derivative of the kernel in Y direction')
    mpl.show()
    return { 'k': k, 'kx': kx ,'ky': ky}

def estimateDensity(
        roi, Lmap, k,
        H, W):
    # print Lmap
    colors = range(1,
        Lmap + 1)
    q = np.zeros((Lmap))
    for i in range(0, H):
        for j in range(0, W):
            q[roi[i][j] + 1] = q[roi[i][j] + 1] + k[i][j]
    # print "Q - : ", len(q)
    # print "Colors: ", len(colors)
    return q


def quantizeImage(centroids,
        roi):
    w, h, d = tuple(roi.shape)
    assert d == 3
    image_array = np.reshape(roi,
        (w * h, d))
    qnt, _ = vq(image_array,
        centroids)
    centers_idx = np.reshape(qnt,
        (roi.shape[0], roi.shape[1]))
    return centers_idx


def colorQuantization(img,
        numColors):
    w, h, d = tuple(img.shape)
    assert d == 3
    image_array = np.reshape(img,
        (w * h, d))
    print("Fitting model on a small sub-sample of the data")
    centroids, _ = kmeans(image_array,
        numColors)
    return centroids

def extractMovieFrames(frames_path,
        video_sequence):
    cap = cv2.VideoCapture(video_sequence)
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)
    elif os.listdir(frames_path) != []:
        files = os.listdir(frames_path)
        for f in files:
            os.remove(frames_path+f)
    frameCount = 1
    while(cap.isOpened()):
        ret, frame = cap.read()
        name = frames_path + "%d.jpg"%frameCount
        cv2.imwrite(name,
            frame)
        if frame is None:
            break
        else:
            frameCount += 1
    return frameCount

def onselect(eclick,
    erelease):
    global x_upper, y_upper, x_lower, y_lower
    x_upper, y_upper, x_lower, y_lower = int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata)
    eclick.button

def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)

def drawRegionOfInterest(img):
    fig1 = mpl.figure(1)
    ax = mpl.subplot(111)
    mpl.imshow(img,
        cmap = cm.Greys_r)
    toggle_selector.RS = RectangleSelector(ax,
        onselect,
        drawtype='box')
    mpl.connect('key_press_event', toggle_selector)
    mpl.show()
    print "Selected ROI shape is: "
    croppedRegion = img[y_upper:y_lower, x_upper:x_lower]
    roiHeight, roiWidth, channels = np.shape(croppedRegion)
    mpl.figure(2)
    mpl.imshow(croppedRegion,
        cmap = cm.Greys_r)
    mpl.suptitle('Chosen ROI')
    mpl.show()
    return { 'roi': croppedRegion, 'H': roiHeight, 'W': roiWidth, 'X0': x_upper ,'Y0': y_upper }

def drawTarget(
        img, x0, y0, H,
        W, savePath, frameIndex):
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    cv2.rectangle(img,
        (x0, y0),
        (x0 + W, y0 + H),
        (74,184,247),
        2)
    cv2.imwrite(savePath
        + str(frameIndex)
        + '.jpg',
        img)
    return

def similarityFunction(
        h1, h2, T2,
        k, H, W):
    w = np.zeros((H,
        W))
    f = 0
    for i in range(0, H):
        for j in range(0, W):
            w[i][j] = np.sqrt(h1[T2[i][j] + 1]
                / float(h2[T2[i][j] + 1]))
            f = f + w[i][j]*k[i][j]
    f = float(f) / (H*W)
    return [f, w]

def performMeanshitTracking(
        x0, y0, f,
        f_indx, loss,
        q, I2, Lmap,
        height, width, f_thresh,
        max_it, H, W,
        k, kx, ky):
    y = y0
    x = x0
    T2 = I2[y:y+H, x:x+W]
    p = estimateDensity(
        T2, Lmap, k,
        H, W)
    step = 1
    [fi, w] = similarityFunction(
        q, p, T2,
        k, H, W)
    f.append(fi)
    while f[f_indx] < f_thresh and step < max_it:
        step += 1
        f_indx += 1
        num_x = 0
        num_y = 0
        den = 0
        for i in range(0, H):
            for j in range(0, W):
                num_x = (num_x
                         + i
                         * w[i][j]
                         * kx[i][j])
                num_y = (num_y
                         + j
                         * w[i][j]
                         * ky[i][j])
                den = (den
                       + w[i][j]
                       * np.linalg.norm([kx[i][j],
                            ky[i][j]]))
        if den != 0:
            dx = np.round(num_x
                          / float(den))
            dy = np.round(num_y
                          / float(den))
            y = y + dy
            x = x + dx
        if (y < 1 or y > height - H) or (x < 1 or x > width - W):
            loss = 1
            break;
        T2 = I2[y:y+H, x:x+W]
        p = estimateDensity(
            T2, Lmap, k,
            H, W)
        [fi, w] = similarityFunction(
            q, p, T2,
            k, H, W)
        f.append(fi)
    return [x, y, loss, f, f_indx]

def checkDirectory(checkPath):
    if not os.path.exists(checkPath):
        os.mkdir(checkPath)
        return
    shutil.rmtree(checkPath)
    return

def createVideo(
    imgPath, videoSeqName,
    codecName, frameRate):
    imgs = natsort.natsort(os.listdir(imgPath))
    sampleImage = cv2.imread(imgPath
                             + imgs[0])
    height , width , dim =  sampleImage.shape
    video = cv2.VideoWriter(
        videoSeqName, codecName, frameRate,
        (width, height), True)
    for eachImage in range(0, len(imgs)):
        print "Currently processing - ", eachImage
        currentFrame = cv2.imread(imgPath
                                  + imgs[eachImage])
        video.write(currentFrame)
    video.release()
    return

def main():
    projectRoot = '/home/Chethan/Desktop/Mean_Shift_Video_Tracking/'
    videoSeq = projectRoot + 'seq2.avi'
    framesPath = projectRoot + 'frames/'
    savePath = projectRoot + 'processedFrames/'
    checkDirectory(framesPath)
    checkDirectory(savePath)
    START_INDEX = 1
    F_THRESHOLD = 0.5
    MAX_ITERATIONS = 10
    KERNEL_RADIUS = 1
    LOSS_VAL = 0
    KERNEL_TYPE = "Gaussian"
    print "Extracting sequences from Video"
    frameCount = extractMovieFrames(framesPath,
        videoSeq)
    print "Sequence extraction complete"
    print "Number of Frames in the Video Sequence: ", frameCount
    img = cv2.imread(framesPath
                     + '1.jpg')
    roiProperties = drawRegionOfInterest(img)
    print "Generating kernel"
    myKernel = kernels(
        roiProperties['H'], roiProperties['W'],
        KERNEL_RADIUS, KERNEL_TYPE)
    mpl.figure(5)
    mpl.imshow(myKernel['k'],
        cmap = cm.Greys_r)
    mpl.suptitle('Chosen kernel')
    mpl.show()
    print "shape of ROI - ", np.shape(roiProperties['roi'])
    print "shape of Kernel - ", np.shape(myKernel['k'])
    print "Performing K-Means on the image"
    kmeansCentroid = colorQuantization(img.astype(float),
        64)
    cmap = kmeansCentroid
    Lmap = len(cmap) + 1;
    print "Displaying the selected target with quantization"
    indexedTarget = quantizeImage(cmap,
        roiProperties['roi'])
    print np.shape(indexedTarget)
    mpl.figure(6)
    mpl.imshow(indexedTarget)
    mpl.show()
    print "Estimating the density of selected target region"
    q = estimateDensity(indexedTarget,
        Lmap, myKernel['k'],
        roiProperties['H'], roiProperties['W'])
    f = list(np.zeros(((frameCount-1)
                        * MAX_ITERATIONS)))
    F_INDEX = 1
    refX, refY = roiProperties['X0'], roiProperties['Y0']
    refH, refW = roiProperties['H'], roiProperties['W']
    drawTarget(
        img, refX, refY, refH,
        refW, savePath, 1)
    print "Tracking the target in remaining frames"
    for t in range(2, frameCount):
        print "Currently processing frame: ", str(t)
        nextFrame = cv2.imread(framesPath
                               + str(t)
                               + '.jpg')
        H, W, D = np.shape(nextFrame)
        nextIndexedFrame = quantizeImage(cmap,
            nextFrame)
        [nX, nY, LOSS_VAL, f, F_INDEX] = performMeanshitTracking(refX, refY, f, F_INDEX, LOSS_VAL,
                                                                 q, nextIndexedFrame, Lmap, H,
                                                                 W, F_THRESHOLD, MAX_ITERATIONS, refH,
                                                                 refW, myKernel['k'], myKernel['kx'], myKernel['ky'])
        if LOSS_VAL == 1:
            print "Lost the target :("
            break
        else:
            print "Target Updated!!"
            drawTarget(
                nextFrame, int(nX), int(nY), int(refH),
                int(refW), savePath, t)
            refY = nY
            refX = nX
    print "Tracking object complete. Now converting sequences into frames"
    createVideo(
        savePath, projectRoot + 'tracking-output.avi',
        cv2.cv.CV_FOURCC('M','J','P','G'), 25)
    print "Conversion done!"


if __name__== "__main__":
    main()
