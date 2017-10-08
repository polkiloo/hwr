"""
  Input a image.
  Output three things:
  1. A binary version of the original image
  2. a binary mask where 1 is text, 0 not
  3. a cleaned version of the original image

"""

import numpy as np
import cv2
import sys
import scipy.ndimage
import argparse
import os

import connected_components as cc
import arg
import defaults


def clean_page(img, max_scale=defaults.CC_SCALE_MAX, min_scale=defaults.CC_SCALE_MIN):
  gray = grayscale(img)

  #create gaussian filtered and unfiltered binary images
  sigma = arg.float_value('sigma',default_value=defaults.GAUSSIAN_FILTER_SIGMA)
  if arg.boolean_value('verbose'):
    print 'Binarizing image with sigma value of ' + str(sigma)

  gaussian_filtered = scipy.ndimage.gaussian_filter(gray, sigma=sigma)
  if arg.boolean_value('verbose'):
    print 'Binarizing image with sigma value of ' + str(sigma)
  gaussian_binary = binarize(gaussian_filtered)
  binary = binarize(gray)

  #Draw out statistics on average connected component size in the rescaled, binary image
  average_size = cc.average_size(gaussian_binary)
  max_size = average_size*max_scale
  min_size = average_size*min_scale

  #primary mask is connected components filtered by size
  mask = cc.form_mask(gaussian_binary, max_size, min_size)

  #secondary mask is formed from canny edges
  canny_mask = form_canny_mask(gaussian_filtered, mask=mask)

  #final mask is size filtered connected components on canny mask
  final_mask = cc.form_mask(canny_mask, max_size, min_size)

  #apply mask and return images
  cleaned = cv2.bitwise_not(final_mask * binary)
  return (cv2.bitwise_not(binary), final_mask, cleaned)

def clean_image_file(filename):
  img = cv2.imread(filename)
  return clean_page(img)

def grayscale(img):
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  return gray

def binarize(img, white=255):
  (t,binary) = cv2.threshold(img, 0, white, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  return binary

def form_canny_mask(img, mask=None):
  edges = cv2.Canny(img, 12, 255, apertureSize=3)
  if mask is not None:
    mask = mask*edges
  else:
    mask = edges
  _, contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

  temp_mask = np.zeros(img.shape,np.uint8)
  for c in contours:
    #also draw detected contours into the original image in green
    #cv2.drawContours(img,[c],0,(0,255,0),1)
    hull = cv2.convexHull(c)
    cv2.drawContours(temp_mask,[hull],0,255,-1)
    #cv2.drawContours(temp_mask,[c],0,255,-1)
    #polygon = cv2.approxPolyDP(c,0.1*cv2.arcLength(c,True),True)
    #cv2.drawContours(temp_mask,[polygon],0,255,-1)
  return temp_mask


if __name__ == '__main__':

  parser = arg.parser
  parser = argparse.ArgumentParser(description='Clean image.')
  parser.add_argument('infile', help='Input (color) image to clean.')
  parser.add_argument('-o','--output', dest='outfile', help='Output (color) cleaned image.')
  parser.add_argument('-b','--binary', dest='binary', default=None, help='Binarized version of input file.')
  parser.add_argument('-v','--verbose', help='Verbose operation. Print status messages during processing', action="store_true")
  parser.add_argument('--display', help='Display output using OPENCV api and block program exit.', action="store_true")
  arg.value = parser.parse_args()

  infile = arg.string_value('infile')
  outfile = arg.string_value('outfile',default_value=infile + '.cleaned.png')
  binary_outfile = arg.string_value('binary',default_value=infile + '.binary.png')
  mask = arg.boolean_value('mask')

  if not os.path.isfile(infile):
    print 'Please provide a regular existing input file. Use -h option for help.'
    sys.exit(-1)

  if arg.boolean_value('verbose'):
    print '\tProcessing file ' + infile
    print '\tGenerating output ' + outfile

  (binary,mask,cleaned) = clean_image_file(infile)

  cv2.imwrite(outfile,cleaned)
  if binary is not None:
    cv2.imwrite(binary_outfile, binary)

  if arg.boolean_value('display'):
    cv2.imshow('Binary',binary)
    cv2.imshow('Cleaned',cleaned)
  if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
  cv2.destroyAllWindows()

