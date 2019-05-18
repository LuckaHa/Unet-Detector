import cv2 as cv
import os
import sys


def thresholdLabeledImages(folder, p):
    for filename in os.listdir(folder):
        name = filename.split('_')
        if len(name) > 1: # len obrazky _predict chceme previest
            img = cv.imread(folder + '/' + filename,0)

            # obrazok ma rozmer 256 x 256
            for i in range(256):
                for j in range(256):
                    if img[i,j] > 256*p: # prah = 256*p
                        img[i,j] = 255
                    else:
                        img[i,j] = 0       
            cv.imwrite(folder + '/' + name[0] + 't.png', img)


# PREMENNE
# nazov priecinka s obrazkami, ktore treba prahovat 
# honota p - kolkopercentne biely pixel ma byt biely
if (len(sys.argv) > 1):
    thresholdLabeledImages(sys.argv[1], float(sys.argv[2])) # OD POUZIVATELA 
else:
    thresholdLabeledImages('data/test/image', 0.9) # DEFAULT
print('Thresholding finished.')

