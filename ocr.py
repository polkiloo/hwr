"""
 Run OCR on some text bounding boxes.

"""
#import clean_page as clean
import connected_components as cc
import run_length_smoothing as rls
import segmentation
import clean_page as clean
import arg
import argparse

import numpy as np
import cv2
import sys
import os
import scipy.ndimage
from pylab import zeros,amax,median

import tesseract

class Blurb(object):
  def __init__(self, x, y, w, h, text, confidence=100.0):
    self.x=x
    self.y=y
    self.w=w
    self.h=h
    self.text = text
    self.confidence = confidence

def draw_2d_slices(img,slices,color=(0,0,255),line_size=1):
  for entry in slices:
    vert=entry[0]
    horiz=entry[1]
    cv2.rectangle(img,(horiz.start,vert.start),(horiz.stop,vert.stop),color,line_size)

def max_width_2d_slices(lines):
  max_width = 0
  for line in lines:
    width = line[1].stop - line[1].start
    if width>max_width:
      width = max_width
  return max_width

def segment_into_lines(img,component, min_segment_threshold=1):
  (ys,xs)=component[:2]
  w=xs.stop-xs.start
  h=ys.stop-ys.start
  x = xs.start
  y = ys.start
  aspect = float(w)/float(h)

  vertical = []
  start_col = xs.start
  for col in range(xs.start,xs.stop):
    count = np.count_nonzero(img[ys.start:ys.stop,col])
    if count<=min_segment_threshold or col==(xs.stop):
      if start_col>=0:
        vertical.append((slice(ys.start,ys.stop),slice(start_col,col)))
        start_col=-1
    elif start_col < 0:
      start_col=col

  horizontal=[]
  start_row = ys.start
  for row in range(ys.start,ys.stop):
    count = np.count_nonzero(img[row,xs.start:xs.stop])
    if count<=min_segment_threshold or row==(ys.stop):
      if start_row>=0:
        horizontal.append((slice(start_row,row),slice(xs.start,xs.stop)))
        start_row=-1
    elif start_row < 0:
      start_row=row
  return (aspect, vertical, horizontal)

def ocr_on_bounding_boxes(img, components):

  blurbs = []
  for component in components:
    (aspect, vertical, horizontal) = segment_into_lines(img, component)

    api = tesseract.TessBaseAPI()
    api.Init(".","rus",tesseract.OEM_DEFAULT)
    if len(vertical)<2:
      api.SetPageSegMode(5)
    else:
      api.SetPageSegMode(tesseract.PSM_AUTO)
    api.SetVariable('chop_enable','T')
    api.SetVariable('use_new_state_cost','F')
    api.SetVariable('segment_segcost_rating','F')
    api.SetVariable('enable_new_segsearch','0')
    api.SetVariable('language_model_ngram_on','0')
    api.SetVariable('textord_force_make_prop_words','F')
    api.SetVariable('tessedit_char_blacklist', '}><L')
    api.SetVariable('textord_debug_tabfind','0')

    x=component[1].start
    y=component[0].start
    w=component[1].stop-x
    h=component[0].stop-y
    roi = cv2.cv.CreateImage((w,h), 8, 1)
    sub = cv2.cv.GetSubRect(cv2.cv.fromarray(img), (x, y, w, h))
    cv2.cv.Copy(sub, roi)
    tesseract.SetCvImage(roi, api)
    txt=api.GetUTF8Text()
    conf=api.MeanTextConf()
    if conf>0 and len(txt)>0:
      blurb = Blurb(x, y, w, h, txt, confidence=conf)
      blurbs.append(blurb)

  return blurbs

def main():
  parser = arg.parser
  parser = argparse.ArgumentParser(description='Basic OCR on image.')
  parser.add_argument('infile', help='Input (color) image to clean.')
  parser.add_argument('-o','--output', dest='outfile', help='Output (color) cleaned image.')
  parser.add_argument('-v','--verbose', help='Verbose operation. Print status messages during processing', action="store_true")
  parser.add_argument('--sigma', help='Std Dev of gaussian preprocesing filter.',type=float,default=None)

  arg.value = parser.parse_args()

  infile = arg.string_value('infile')
  outfile = arg.string_value('outfile', default_value=infile + '.html')

  if not os.path.isfile(infile):
    print 'Please provide a regular existing input file. Use -h option for help.'
    sys.exit(-1)

  if arg.boolean_value('verbose'):
    print '\tProcessing file ' + infile
    print '\tGenerating output ' + outfile

  img = cv2.imread(infile)
  gray = clean.grayscale(img)
  binary = clean.binarize(gray)

  segmented = segmentation.segment_image_file(infile)

  components = cc.get_connected_components(segmented)

  blurbs = ocr_on_bounding_boxes(binary, components)
  for blurb in blurbs:
    print str(blurb.x)+','+str(blurb.y)+' '+str(blurb.w)+'x'+str(blurb.h)+' '+ str(blurb.confidence)+'% :'+ blurb.text


if __name__ == '__main__':
  main()

