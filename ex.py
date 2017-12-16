import numpy as np
import cv2
import sys
import argparse
import os

import arg
def nothing(x):
    pass

def main():
  parser = arg.parser
  parser = argparse.ArgumentParser(description='Segment image.')
  parser.add_argument('infile', help='Input (color) image to clean.')
  parser.add_argument('--display', help='Display output using OPENCV api and block program exit.', action="store_true")

  arg.value = parser.parse_args()

  infile = arg.string_value('infile')
  outfile = arg.string_value('outfile', default_value=infile + '.segmented.png')
  binary_outfile = infile + '.binary.png'

  if not os.path.isfile(infile):
    print 'Please provide a regular existing input file. Use -h option for help.'
    sys.exit(-1)

  if arg.boolean_value('verbose'):
    print '\tProcessing file ' + infile
    print '\tGenerating output ' + outfile

  cv2.namedWindow('example')


  img = cv2.imread(infile)
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  vis = img.copy()
  #Create MSER object
  mser = cv2.MSER_create()
  #detect regions in gray scale image
  regions, _ = mser.detectRegions(gray)

  hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

  cv2.imshow('img', vis)

  mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

  for contour in hulls:

      cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
  print mask

  #this is used to find only text regions, remaining are ignored
  text_only = cv2.bitwise_and(img, img, mask=mask)

  cv2.imshow("text only", text_only)
  height, width = gray.shape

  cv2.createTrackbar('Threshold windows size','example',0, width / 3, nothing)
  cv2.createTrackbar('Gauss kernel size','example',0, width / 3, nothing)
  while 1:
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
      break

    tws = cv2.getTrackbarPos('Threshold windows size','example')
    gks = cv2.getTrackbarPos('Gauss kernel size','example')

    kernel = np.ones((1,gks+1),np.float32)/(gks+1)
    thresh = cv2.adaptiveThreshold(gray , 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,\
     tws*2 + 3, 255 - img.mean())

    filtered = cv2.filter2D(thresh, -1, kernel)
    edges = cv2.Canny(filtered, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 255, 7)
    im2, contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #contours = sorted(contours, key = cv2.contourArea,reverse = True)[:100]
    dci = np.copy(img)
    cv2.drawContours(dci, contours, -1, (0,255,0), 3)

    cv2.imshow('example',thresh)


  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
