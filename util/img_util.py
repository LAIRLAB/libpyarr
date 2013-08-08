#3rd party modules
import os, Image, ImageDraw, numpy as np, scipy.misc
import scipy.ndimage
import matplotlib.pyplot as plt
from pdbwrap import *
import signal

import color_printer as cpm
import verify_util as vu
import type_util as tu
import rand_util as ru
import pdb
import subprocess

try:
    from PySide.QtGui import QImage as QImage, QPixmap as QPixmap
    qt_imported = True
except ImportError:
    pass

import numpy

im_suffixes = ['.jpg', '.png', '']

def load_basename(prefix, mode='RGB'):
    im = None
    imfname = None
    for suffix in im_suffixes:
        imfname = prefix + suffix

        try:
            im = Image.open(imfname)
            if im.mode != 'RGB':
                im = None
                continue
        except IOError:
            continue

    if im is None:
        raise IOError("Image for basename: {} could not be loaded as RGB".format(prefix))
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

def overlay_classification(image, labels, colormap, alpha_im = 0.4, alpha_color = 0.6,
                           image_view_map = None):
    if image_view_map is None:
        image_view = image
        labels_view = labels
    else:
        image_view = image[image_view_map]
        labels_view = labels[image_view_map]
    image[image_view_map] = numpy.uint8(alpha_im * image_view + alpha_color * colormap[labels_view])
    return image

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
 
# def overlay_bboxes(pil_im, bboxes_alg_dict, **kwargs):
#     outlines = kwargs.get('outlines', ['blue' for b in range(len(bboxes_alg_dict.keys()))])
#     thickness = kwargs.get('thickness', 2)
#     for (key, n) in zip(bboxes_alg_dict.keys(), range(len(bboxes_alg_dict.keys()))):
#         for b in bboxes_alg_dict[key]:
#             cn = b.get_corners()
#             ImageDraw.Draw(pil_im).rectangle(b.get_corners(), outline = outlines[key], fill = None)
#             for t in range(thickness):
#                 c2 = [(cn[0][0] + t, cn[0][1] + t),  (cn[1][0] + t, cn[1][1] + t)]
#                 ImageDraw.Draw(pil_im).rectangle(c2, outline = outlines[key], fill = None)
#     return pil_im

def overlay_bboxes(pil_im, bboxes):
    for b in bboxes:
        pil_im = overlay_bbox(pil_im, b)
    return pil_im

def overlay_bbox(pil_im, bbox, **kwargs):
    cn = pil_im.__class__.__name__
    if cn in ['Image', 'PngImageFile', 'JpegImageFile']:
        pass
    elif cn == 'ndarray':
        if pil_im.ndim == 2:
            #pil_im *= 255
            pil_im = numpy.uint8(numpy.dstack((pil_im, pil_im, pil_im)))
        pil_im = Image.fromarray(pil_im)
    else:
        raise RuntimeError("Can't deal with image of type: {}".format(cn))

    c = kwargs.get('outline', 'red')
    
    if hasattr(bbox, 'get_corners'):        
        nonshitty_rectangle(ImageDraw.Draw(pil_im), bbox.get_corners)
    elif hasattr(bbox, 'len') and len(bbox) == 4:
        nonshitty_rectangle(ImageDraw.Draw(pil_im), bbox)
    elif tu.hasattrs(bbox, ['x', 'y', 'width', 'height']):
        bbox_coords = (bbox.x, bbox.y, 
                       bbox.x + bbox.width, bbox.y + bbox.height)
        nonshitty_rectangle(ImageDraw.Draw(pil_im), bbox_coords)
    else:
        raise ValueError("Unsupported bounding box type")
    return numpy.asarray(pil_im).copy()

def nonshitty_rectangle(draw_inst, bbox, width = 2, outline = 'blue'):
    p1 = (bbox[0], bbox[1])
    p2 = (bbox[0], bbox[3])
    p3 = (bbox[2], bbox[1])
    p4 = (bbox[2], bbox[3])
    draw_inst.line([p1, p2], width = width, fill = outline)
    draw_inst.line([p1, p3], width = width, fill = outline)
    draw_inst.line([p2, p4], width = width, fill = outline)
    draw_inst.line([p3, p4], width = width, fill = outline)    

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
        raise ValueError
    
    ret = numpy.array(Image.fromarray(arr, mode=mode).resize(new_size, interp), copy = True)

    return ret

def pil_to_pixmap(im):
    data = im.convert('RGBA').tostring('raw', 'BGRA')
    image = QImage(data, im.size[0], im.size[1], QImage.Format_ARGB32)
    return QPixmap.fromImage(image)
    
def rasterize_numpy(np):
    mi = np.min()
    ma = np.max()
    colormap = []
    if np.ndim == 3:
        return Image.fromarray(np)
    if np.max() <= 1.0:
        return rasterize_probmap(np)
    for i in range(mi, ma + 1):
        mapping = [i]
        mapping.extend(ru.random_8bit_rgb())
        colormap.append(mapping)
    return map_ascii_to_pil(np, colormap)

def rasterize_objects(shape, objs):
    z = numpy.zeros(shape, dtype = numpy.uint8)
    for o in objs:
        z[o] = 1
    return rasterize_numpy(z)
    

def rasterize_probmap(pm):
    if pm.max() <= 1.0:
        return Image.fromarray(numpy.uint8(255 * pm))
    elif pm.max() <= 255:
        return Image.fromarray(numpy.uint8(pm))
    else:
        raise RuntimeError(cpm.gcp.error("unknown probmap type"))
    

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
            color = colormap[ascii_array[i,j]][1:]
            try:
                im[i,j] = [color.r(), color.g(), color.b()]
            except AttributeError:
                im[i,j] = color
            
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


def non_max_suppression_2d(data, ws = 5):
    nms = numpy.zeros(data.shape)

    for r in range(ws, data.shape[0] - ws):
        for c in range(ws, data.shape[1] - ws):
            v = data[r][c]
            try:
                region_max = data[r - ws : r + ws, c - ws : c + ws].max()
                if v >= region_max:
                    nms[r][c] = v
            except IndexError:
                continue
    return nms
         
#takes a labeled map (e.g. a segmentmap)       
def get_centroids_map(np_map):
    centroids_map = numpy.zeros(np_map.shape[:2], dtype = numpy.uint8)
    centroids = []
    for s in range(np_map.min(), np_map.max()):
        center = [int(x.mean()) for x in numpy.where(np_map == s) if x.size > 0]
        if len(center) == 0:
            centroids.append((s, [None, None]))
            continue
        else:
            centroids.append((s, center))
        centroids_map[center[0], center[1]] = 255
    return centroids_map, centroids

def get_segment_borders(np_map, val = 255):
    npa = numpy.zeros((np_map.shape[0], np_map.shape[1]))
    
    num_segments = np_map.max() - np_map.min() + 1
    perimeter_pixels = [[] for x in range(num_segments)]
    
    flag_counted = npa.copy()
    
    rows = np_map.shape[0]
    cols = np_map.shape[1]

    #modeled after SegmentMap::computeBorders
    for r in range(rows):
        for c in range(cols):
            segment_a = np_map[r][c]

            if (r == 0 or c == 0 or r == (rows - 1) or c == (cols - 1)):
                if (not flag_counted[r][c]):
                    flag_counted[r][c] = 1
                    perimeter_pixels[segment_a].append(r * cols + c)
                    npa[r][c] = val

            #right
            if (c < (cols - 1)):
                r_r = r
                c_r = c + 1
                segment_r = np_map[r_r][c_r]
                if (segment_a != segment_r):
                    if (not flag_counted[r][c]):
                        flag_counted[r][c] = 1
                        perimeter_pixels[segment_a].append(r * cols + c)
                        npa[r][c] = val

                    if (not flag_counted[r_r][c_r]):
                        flag_counted[r_r][c_r] = 1
                        perimeter_pixels[segment_r].append(r_r * cols + c_r)
                        npa[r_r][c_r] = val
            
            #down-left, down, down-right
            if (r < (rows - 1)):
                
                #down-left
                if c > 0:
                    r_dl = r + 1
                    c_dl = c - 1
                    segment_dl = np_map[r_dl][c_dl]
                    if (segment_a != segment_dl):
                        if (not flag_counted[r][c]):
                            flag_counted[r][c] = 1
                            perimeter_pixels[segment_a].append(r * cols + c)
                            npa[r][c] = val

                        if (not flag_counted[r_dl][c_dl]):
                            flag_counted[r_dl][c_dl] = 1
                            perimeter_pixels[segment_dl].append(r_dl * cols + c_dl)
                            npa[r_dl][c_dl] = val

                r_d = r + 1
                c_d = c
                segment_d = np_map[r_d][c_d]

                #down
                if  (segment_a != segment_d):
                    if (not flag_counted[r][c]):
                        flag_counted[r][c] = 1
                        perimeter_pixels[segment_a].append(r * cols + c)
                        npa[r][c] = val

                    if (not flag_counted[r_d][c_d]):
                        flag_counted[r_d][c_d] = 1
                        perimeter_pixels[segment_d].append(r_d * cols + c_d)
                        npa[r_d][c_d] = val

                #down-right
                if (c < cols - 1):
                    r_dr = r + 1
                    c_dr = c + 1
                    segment_dr = np_map[r_dr][c_dr]
                    if (segment_a != segment_dr):
                        if (not flag_counted[r][c]):
                            flag_counted[r][c] = 1
                            perimeter_pixels[segment_a].append(r * cols + c)
                            npa[r][c] = 1

                        if (not flag_counted[r_dr][c_dr]):
                            flag_counted[r_dr][c_dr] = 1
                            perimeter_pixels[segment_dr].append(r_dr * cols + c_dr)
                            npa[r_dr][c_dr] = 1

            #end down
        #end cols
    #end rows
    return npa
                    
                    
#dont use this it has a hack          
class IntegralImage(object):
    def __init__(self, x):
        self.ii = x.cumsum(1).cumsum(0)

    def integrate(self, r0, c0, r1, c1):
        S = 0
        S += self.ii[r1, c1]

        if (r0 - 1 >= 0) and (c0 - 1 >= 0):
            S += self.ii[r0 - 1, c0 - 1]

        if (r0 - 1 >= 0):
            S -= self.ii[r0 - 1, c1]

        if (c0 - 1 >= 0):
            S -= self.ii[r1, c0 - 1]

        return S

    def integrate_box(self, b):
        r0 = int(b.y)
        c0 = int(b.x)
        
        #hack 
        r1 = min(319, int(r0 + b.height))
        c1 = min(319, int(c0 + b.width))
        return self.integrate(r0, c0, r1, c1)

def clip_bboxes(bboxes, shape):
    for bbox in bboxes:
        bbox.x = min(max(bbox.x, 0), shape[1]-1)
        bbox.y = min(max(bbox.y, 0), shape[0]-1)
        if bbox.x + bbox.width < shape[1]:
            pass
        else:
            bbox.width = shape[1] - bbox.x - 1

        if bbox.y + bbox.height < shape[0]:
            pass
        else:
            bbox.height = shape[0] - bbox.y - 1

def segments_adjacent(segmap1, segmap2):
    (la, num_labels) = scipy.ndimage.label(\
        numpy.logical_or(segmap1, segmap2))
    return num_labels < 2

class BoundingBox(object):
    def __init__(self, x, y, width, height, score = None, type = None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.score = score
        self.type = type

    def contains(self, x, y):
        if x >= self.x and y >= self.y:
            if x <= self.x + self.width and y <= self.y + self.height:
                return True
        return False

    def npy_binary(self, shape):
        n = numpy.zeros(shape, dtype=numpy.uint8)
        n[self.y : self.y + self.height,
          self.x : self.x + self.width] = 1
        return n

    def boundaries(self):
        return [self.y, self.y + self.height, self.x, self.x + self.width]
    

def boundaries_npy(arr):
    a = numpy.argwhere(arr)
    (y_min, x_min), (y_max, x_max) = a.min(0), a.max(0) + 1
    return (y_min, x_min), (y_max, x_max)

def bounding_box_npy(arr):
    (y_min, x_min), (y_max, x_max) = boundaries_npy(arr)
    return BoundingBox(x = x_min, 
                       y = y_min,
                       width = x_max - x_min,
                       height = y_max - y_min)

def logical_centroid(arr):
    if numpy.count_nonzero(arr) == 0:
        return (None, None)
    a = numpy.argwhere(arr)
    (y_min, x_min), (y_max, x_max) = a.min(0), a.max(0) + 1
    width = x_max - x_min
    height = y_max - y_min
    return (y_min + int(height / 2.0), x_min + int(width / 2.0))
    
#expects a labeled array
def find_noncontiguous_objects(arr, ignore = [0], min_matching = 0):
    assert(arr.size > 0)
    assert(arr.dtype == numpy.uint8)
    
    ma = arr.max()
    objects = []
    for i in numpy.unique(arr):
        if i in ignore:
            continue

        o = arr == i
        if numpy.count_nonzero(o) > min_matching:
            objects.append(o)
    return objects
    
#expects a labeled array
def remove_small_regions(arr, min_size = 10, ignore = [0]):
    assert(arr.dtype == numpy.uint8)
    for val in numpy.unique(arr):
        if val in ignore:
            continue
        mask = arr == val
        if numpy.count_nonzero(mask) < min_size:
            arr[mask] = 0
    
# def render_3_chunks(im, chunks):
#     assert(len(chunks) == 3):
#     for c in chunks:
        

#returns 2^its  morphings
def dilation_erosion_tree(arr, its = 2):
    assert(its >= 1)
    r = []
    e = scipy.ndimage.binary_erosion(arr)
    d = scipy.ndimage.binary_dilation(arr)
    if its == 1:
        return [d, e]
    else:
        r.extend(dilation_erosion_tree(d, its - 1))
        r.extend(dilation_erosion_tree(e, its - 1))
        return r
