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

  arg.value = parser.parse_args()

  infile = arg.string_value('infile')
  outfile = arg.string_value('outfile', default_value=infile + '.segmented.png')

  cv2.namedWindow('text only')
  if not os.path.isfile(infile):
    print 'Please provide a regular existing input file. Use -h option for help.'
    sys.exit(-1)

  while 1:
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
      break

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

    #this is used to find only text regions, remaining are ignored
    text_only = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("text only", text_only)
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
