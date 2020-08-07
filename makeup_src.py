import cv2,sys,dlib,time,math
import numpy as np
import faceBlendCommon as fbc
import mls as mls
import tempfile
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = (8.0,8.0)
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['image.interpolation'] = 'bilinear'

# initial color
#color = (0,0,0)

#input
#argv = sys.argv[1:]
#im_path, option = argv[:2]

# start time
#start = time.time()

# convert data type
#option = (int)(option)
#im_path = (str)(im_path)

#tmp_name = tempfile.NamedTemporaryFile()

# Landmark model location
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
#OUTPUT_PATH = tempfile.mkstemp('.png', 'makeup', '/tmp')[1]
OUTPUT_PATH = 'static/image/output.png'
# Get the face detector
faceDetector = dlib.get_frontal_face_detector()
# The landmark detector is implemented in the shape_predictor class
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

def lip_makeup(im, color):
    # Read image
    #im = cv2.imread(im_path) 

    imDlib = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #img1Warped = np.copy(imDlib)

    landmarks = fbc.getLandmarks(faceDetector, landmarkDetector, imDlib)

    #selectedIndex_upper_lip  = [ 48, 49, 50, 51, 52, 53, 54, 61, 62, 63]
    #selectedIndex_botto_lip  = [ 55, 56, 57, 58, 59, 60, 64, 65, 66, 67]
    selectedIndex_upper_lip  = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    selectedIndex_botto_lip  = [60, 61, 62, 63, 64, 65, 66, 67]


    hull1, hull2 = [], []

    for i in selectedIndex_upper_lip:
        hull1.append(landmarks[i])
        
    for j in selectedIndex_botto_lip:
        hull2.append(landmarks[j])


    mask1 = np.zeros(imDlib.shape, dtype=imDlib.dtype)
    cv2.fillConvexPoly(mask1, np.array([hull1], dtype=np.int32), color)

    mask2 = np.zeros(imDlib.shape, dtype=imDlib.dtype)
    cv2.fillConvexPoly(mask2, np.array([hull2], dtype=np.int32), color)

    mask = cv2.addWeighted(mask1, 1, mask2, 1, 0.0)

    # Blurring face mask to alpha blend to hide seams
    #kernel = np.ones((10,10),np.uint8)
    maskHeight, maskWidth = mask.shape[0:2]
    maskSmall = cv2.resize(mask, (256, int(maskHeight*256.0/maskWidth)))
    maskSmall = cv2.erode(maskSmall, (-1, -1), 5)
    #maskSmall = cv2.erode(maskSmall, kernel, 1)
    maskSmall = cv2.GaussianBlur(maskSmall, (11, 11), 0, 0)
    mask = cv2.resize(maskSmall, (maskWidth, maskHeight))

    feature_image = cv2.addWeighted(mask, 0.1, imDlib, 1, 0.0)
    #displayImage = np.hstack((imDlib, feature_image))

    feature_image = cv2.cvtColor(feature_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(OUTPUT_PATH, feature_image)

    # end time
    #cc = time.time() - start
    return OUTPUT_PATH


#print(cc)
#cv2.imwrite("abinh.jpg", feature_image)

#plt.figure(figsize = (12, 12))
#plt.imshow(displayImage)
#plt.show()

def blush_makeup(im, color):
    #im = cv2.imread(im_path) 

    imDlib = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #img1Warped = np.copy(imDlib)

    landmarks = fbc.getLandmarks(faceDetector, landmarkDetector, imDlib)

    #selectedIndex_upper_lip  = [  1 , 2 , 3 , 4 , 31, 40, 41]
    #selectedIndex_botto_lip  = [  13, 14, 15, 16, 35, 46, 47]

    selectedIndex_upper_lip  = [  1 , 2 , 3 , 4 , 31, 40, 41]
    selectedIndex_botto_lip  = [  13, 14, 15, 16, 35, 46, 47]


    hull1, hull2 = [], []

    for i in selectedIndex_upper_lip:
        hull1.append(landmarks[i])
        
    for i in selectedIndex_botto_lip:
        hull2.append(landmarks[i])


    mask1 = np.zeros(imDlib.shape, dtype=imDlib.dtype)
    cv2.fillConvexPoly(mask1, np.array([hull1], dtype=np.int32),  color)

    mask2 = np.zeros(imDlib.shape, dtype=imDlib.dtype)
    cv2.fillConvexPoly(mask2, np.array([hull2], dtype=np.int32),  color)

    mask = cv2.addWeighted(mask1, 1, mask2, 1, 0.0)

    # Blurring face mask to alpha blend to hide seams
    maskHeight, maskWidth = mask.shape[0:2]
    maskSmall = cv2.resize(mask, (256, int(maskHeight*256.0/maskWidth)))
    maskSmall = cv2.erode(maskSmall, (-1, -1), 20)
    maskSmall = cv2.GaussianBlur(maskSmall, (51, 51), 0, 0)
    mask = cv2.resize(maskSmall, (maskWidth, maskHeight))

    feature_image = cv2.addWeighted(mask, 0.1, imDlib, 0.98, 0.0)
    #displayImage = np.hstack((imDlib, feature_image))

    feature_image = cv2.cvtColor(feature_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(OUTPUT_PATH, feature_image)

    return OUTPUT_PATH