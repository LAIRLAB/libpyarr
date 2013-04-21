#3rd party modules
import os, Image, ImageDraw, numpy as np, scipy.misc
from pdbwrap import *

import color_printer as cpm
import verify_util as vu

try:
    from PySide.QtGui import QImage as QImage, QPixmap as QPixmap
    qt_imported = True
except ImportError:
    pass

import numpy

im_suffixes = ['.jpg', '.png', '']

def open_zero_one(fn):
    assert(os.path.isfile(fn))
    I_np = numpy.asarray(Image.open(fn), dtype = numpy.float64) / 255.0
    return I_np

def load_basename(prefix, mode='RGB'):
    im = None
    imfname = None
    for suffix in im_suffixes:
        imfname = prefix + suffix
        try:
            im = Image.open(imfname)
            if im.mode != 'RGB':
                raise IOError("Image is not RGB")
        except IOError:
            continue
    if im is None:
        raise IOError("Image for basename: {} could not be loaded".format(prefix))
    return (numpy.array(im).copy(), imfname)

def get_segmentwise_distribution(seg, probs):
    sorted_segs, unique_seg_inds = numpy.unique(seg, return_index = True)
    pd = numpy.reshape(probs, (probs.shape[0] * probs.shape[1],
                                       probs.shape[2]))
    seg_dist = pd[unique_seg_inds, :]
    vu.verify_2d_distribution(seg_dist, len(sorted_segs), probs.shape[2])
    return seg_dist

def get_best_labels(segmentation, segmentwise_probabilities):
    prob_map = segmentwise_probabilities[segmentation]
    print "prob map shape: {}".format(prob_map.shape)
    best_labels = numpy.uint8(prob_map.argmax(axis=2))
    return best_labels

def overlay_best_classification(image, segmentation, probabilities, colormap, alpha_im = .4, alpha_color = .6):
    best_labels = get_best_labels(segmentation, probabilities)
    return overlay_classification(image, best_labels, colormap, alpha_im, alpha_color)

def overlay_classification(image, labels, colormap, alpha_im = 0.4, alpha_color = 0.6):
    return numpy.uint8(alpha_im * image + alpha_color * colormap[labels])

def probmap_to_heatmap(image_shape, prob_map, label_idx):
    hm = numpy.uint8(255*numpy.reshape(prob_map[:, label_idx], image_shape))
    return hm

def bulk_heatmap_overlay(images, heatmaps, out_fns, output_dir = 'bulk_overlayed', alpha_im = .4, alpha_color = .6, max_norm = True):
    assert(alpha_im + alpha_color == 1)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if max_norm:
        m = numpy.asarray(heatmaps, dtype = numpy.float64).max()
        cpm.gcp.info("using max to normalize: {}".format(m))
    else:
        m = 255

    for (im, hm, out_fn) in zip(images, heatmaps, out_fns):
        cpm.gcp.debug("Heatmap stats: max: {}, min: {}".format(hm.max(), hm.min()))
        null_channel = numpy.zeros(hm.shape)
        overlayed = numpy.uint8(alpha_im * im + numpy.dstack((null_channel, null_channel, alpha_color * (hm / m))))
        Image.fromarray(overlayed).save('{}/{}'.format(output_dir, out_fn))

def bulk_overlay(images, labels, colormap, out_fns, output_dir = 'bulk_overlayed', alpha_im = .4, alpha_color = .6):
    assert(alpha_im + alpha_color == 1)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for (im, label, out_fn) in zip(images, labels, out_fns):
        overlayed = overlay_classification(im, label, colormap, alpha_im, alpha_color)
        Image.fromarray(overlayed).save('{}/{}'.format(output_dir, out_fn))
 
def overlay_bboxes(pil_im, bboxes_alg_dict, **kwargs):
    outlines = kwargs.get('outlines', ['blue' for b in range(len(bboxes_alg_dict.keys()))])
    thickness = kwargs.get('thickness', 2)
    for (key, n) in zip(bboxes_alg_dict.keys(), range(len(bboxes_alg_dict.keys()))):
        for b in bboxes_alg_dict[key]:
            cn = b.get_corners()
            ImageDraw.Draw(pil_im).rectangle(b.get_corners(), outline = outlines[key], fill = None)
            for t in range(thickness):
                c2 = [(cn[0][0] + t, cn[0][1] + t),  (cn[1][0] + t, cn[1][1] + t)]
                ImageDraw.Draw(pil_im).rectangle(c2, outline = outlines[key], fill = None)
    return pil_im

def overlay_bbox(pil_im, bbox, **kwargs):
    c = kwargs.get('outline', 'blue')
    ImageDraw.Draw(pil_im).rectangle(bbox.get_corners())
    return pil_im

def imresize(arr, size=None, **kwargs):
    interp = kwargs.get('interp', Image.BILINEAR)
    verb = kwargs.get('verb', 'info')
    scale = kwargs.get('scale', None)
    mode = kwargs.get('mode', None)
    
    if scale is not None and scale > 0:
        size = (int(round(arr.shape[0]*scale)),
                int(round(arr.shape[1]*scale)))

    if size is not None:
        if isinstance(size, tuple):
            new_size = (size[1], size[0])
        else:
            maxdim = size
            if arr.shape[0] < arr.shape[1]:
                new_size = (maxdim,
                            int(round(maxdim*(arr.shape[0]*1.0/arr.shape[1]))))
            else:
                new_size = (int(round(maxdim*(arr.shape[1]*1.0/arr.shape[0]))),
                            maxdim)
    else:
        print "Gotta specify a size or a scale to imresize."
        raise ArgumentError
    
    ret = numpy.array(Image.fromarray(arr, mode=mode).resize(new_size, interp), copy = True)

    return ret

def pil_to_pixmap(im):
    data = im.convert('RGBA').tostring('raw', 'BGRA')
    image = QImage(data, im.size[0], im.size[1], QImage.Format_ARGB32)
    return QPixmap.fromImage(image)
    
#requires a square integer image whose indices map to colors in a colormap object
#the colormap is map of format:
# <index> : [label, Color]
#where Color has r,g,b, values
def map_ascii_to_pil(ascii_array,colormap):
    rows=len(ascii_array)
    cols=len(ascii_array[0])
    im = numpy.zeros((rows,cols,3),dtype='uint8')
    
    for i in range(rows):
        for j in range(cols):
            color = colormap[ascii_array[i,j]][1]
            im[i,j] = [color.r(), color.g(), color.b()]
            
    pil_image = Image.fromarray(im,'RGB')
    return pil_image

#converts a number between xmin and xmax (so, grayscale) to be 
#an [r,g,b] color in JET colormap. By default, assumes
#8-bit color, so multiplies by 255
def get_jet_color(x, xmin, xmax, new_min=0.0,new_max=255.0):
    
    jet = [1.0, 1.0, 1.0]
    if(x < xmin): x = float(xmin)
    elif(x > xmax): x = float(xmax)
    dx = float(xmax - xmin)
    
    if(x < (xmin + .25*dx)):
        jet[0] = new_min
        jet[1] = 4 * (x-xmin)/dx;
        
    elif (x < (xmin + .5*dx)):
        jet[0] = new_min
        jet[2] = 1 + 4*(xmin + .25*dx - x)/dx
        
    elif (x < (xmin + .75*dx)):
        jet[0] = 4*(x-xmin-.5*dx)/dx
        jet[2] = new_min
        
    else:
        jet[1] = 1 + 4*(xmin+.75*dx-x)/dx
        jet[2] = new_min
        
    return [c*new_max for c in jet]

#2d to 3d transformation (MxN) -> (MxNx3)
def map_pdf_to_jet(ascii_array):
    rows = len(ascii_array)
    cols = len(ascii_array[0])
    im = numpy.zeros((rows,cols,3),dtype='uint8')
    for i in range(rows):
        for j in range(cols):
            color = get_jet_color(ascii_array[i,j],0.0,1.0,0.0,255.0)
            im[i,j] = color
    return Image.fromarray(im,'RGB')

#look through CvSeq
def get_good_level(level):
    if level != None:
        level = level.h_next()                    
        if(level == None or len(level)<2):
            return 0,False
        else:
            return level,True
    else:
        return 0,False

def resize_image_to_cols(image,new_cols):
    size = image.size

    ar = size[1]/(1.0*size[0])
    new_rows = int(ar * new_cols)
    new_size = [new_cols,new_rows]

    return image.resize(new_size)
    
def get_ar_maintained_size(size, new_max):
    if size[0] > size[1]:
        return [new_max, int(new_max*(size[1]*1.0/size[0]))]
    else:
        return [int(new_max*(size[0]*1.0/size[1])), new_max]

def open_pfm(filename):

    return pdbwrap(open_pfm_are_you_kidding_me_why_does_this_format_exist)(filename)

def open_pfm_are_you_kidding_me_why_does_this_format_exist(filename):
    import struct


    fh = open(filename, 'r')
    id_line = fh.readline().strip()
    dimensions_line = fh.readline().strip().split()
    sf_line = fh.readline().strip()
    data = fh.readline()

    width = int(dimensions_line[0])
    height = int(dimensions_line[1])
    im = numpy.zeros((height, width))
   # print id_line
   # if id_line is not 'Pf':
   #     raise TypeError("Cannot read non-grayscale pfm")

    for r_idx in range(0, height):
        for c_idx in range(0, width):
            print "r_idx, c_idx, data:", r_idx, c_idx, data[4*(c_idx*r_idx+c_idx) : 4*(c_idx*r_idx + c_idx + 1)]
            im[r_idx, c_idx] = struct.unpack('f', data[4*(c_idx*r_idx+c_idx) : 4*(c_idx*r_idx + c_idx + 1)])[0]
    return im

