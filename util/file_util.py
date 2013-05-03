import color_printer as cpm
import os, sys
import type_util
import datetime
import detector_core.ta2_globals as ta2_globals
import Image, numpy
import img_util

def savez_dir(dirname, arr):
    try:
        os.system('rm -rf %s'%dirname)
    except:
        pass
    os.mkdir(dirname)

    size = 1
    for i in xrange(len(arr.shape)):
        size *= arr.shape[i]

    n_slices = size * 8 / 2**31 + 1

    for i in xrange(n_slices):
        numpy.savez_compressed(dirname + '/arr_%d.npz'%i, 
                               arr[i*arr.shape[0]/n_slices : 
                                   (i+1)*arr.shape[0]/n_slices, 
                                   ...])
def loadz_dir(dirname):
    files = [f for f in os.listdir(dirname) if f[-4:] == '.npz']
    arrfiles = [d + '/arr_%d.npz'%i for i in xrange(len(files))]
    
    if len(arrfiles) != len(files):
        print "oh no there's crap in a savearr dir:", dirname

    return numpy.concatenate([numpy.load(f)['arr_0'] for f in arrfiles], 
                             axis=0)

def path_relative_to_file(file_path, path):
    return os.path.abspath(os.path.dirname(file_path) + '/' + path)

def basename_split(bname):
    return bname.strip('/').split('/')[-1].split('.')[0]

def gts(m = True, s = False, ms = False):
    return generate_timestamp(m, s, ms)

def generate_timestamp(minutes=True, seconds=False, microsecond = False):
    stamp = ""
    now = datetime.datetime.now()
    stamp += str(now.month)+"-"+str(now.day)+"-"+str(now.hour)
    if minutes:
        stamp += "-"+str(now.minute)
    if seconds:
        stamp += "-"+str(now.second)
    if microsecond:
        stamp += '-' + str(now.microsecond)
    return stamp

def verify_json_list(json_files, n_required):
    if len(json_files) != n_required:
        cpm.gcp.error("Wrong number of json files! (wanted {}) Failure."
                 .format(n_required))
        return False
    cpm.gcp.info("Checking for JSON existence of {} files".format(n_required))
    for j in json_files:
        if not os.path.isfile(j):
            cpm.gcp.error("JSON file: '{}' doesn't exist.".format(j))
            return False
    return True
     
def load_basenames(basenames_fn):
    return [b.strip() for b in open(basenames_fn).readlines()]
   
def load_segmentation(fname):
    return numpy.genfromtxt(fname, dtype = numpy.uint8)

def load_segment_probabilities(fname):
    return numpy.genfromtxt(fname, dtype = numpy.float64)

def verify_basenames(basenames, im_dir, gt_dir):
        num_bad = 0
        cpm.gcp.info("Checking images and GT for {} basenames...".format(len(basenames)))
        for b in basenames:
            cpm.gcp.debug("checking {}".format(b))
            im_loaded = False
            gt_loaded = False
            for s in img_util.im_suffixes:
                im_fn = '{}/{}{}'.format(im_dir, b, s)
                try:
                    im_loaded = Image.open(im_fn).mode == 'RGB'
                    break
                except IOError:
                    pass
            if not im_loaded:
                cpm.gcp.error("basename: {} had bad image. (no RGB image existed for suffixes: {})".format(b, img_util.im_suffixes))
            try:
                load_integer_map('{}/{}'.format(gt_dir, b))
                gt_loaded = True
            except IOError:
                cpm.gcp.error("basename: {} had bad gt".format(b))
            if not im_loaded or not gt_loaded:
                num_bad += 1
                basenames.remove(b)
        return num_bad
    
#load txt file from .txt or .npz
def load_integer_map(prefix):
    import numpy
    try:
        with open(prefix + '.npz') as f:
            a = numpy.load(f)['arr_0']
            return a
    except IOError:
        try:
            with open(prefix + '.txt') as f:
                a = numpy.genfromtxt(f, dtype=ta2_globals.ground_truth_dtype)
                return a
        except IOError:
            s = "No integer map {}(.npz, .txt) found".format(prefix)
            raise IOError(s)

def require_existence(f, quiet=False):
    if isinstance(f,list):
        for d in f:
            require_existence(d)
    elif isinstance(f,str):
        try:
            with open(f) as test:
                test.close()
                return f
        except:
            if os.path.isdir(f):
                return
            else:
                e ="File '{}' doesn't exist.".format(f)
                if(quiet):
                    raise IOError(e)
                else:
                    raise IOError(cpm.p(e,'r'))
    elif isinstance(f,unicode):
        require_existence(f.encode('ascii','ignore'))
    else:
        raise TypeError("Unknown type to check existence of: {}".format(type(f)))
                        
def desire_existence(f,quiet=False):
    if isinstance(f,list):
        for d in f:
            desire_existence(d)
    elif isinstance(f,str):
        try:
            with open(f) as test:
                test.close()
                return True
        except:
            if os.path.isdir(f):
                return True
            else:
                #print warning in yellow
                if not quiet:
                    cpm.p("File '{}' doesn't exist.".format(f),'y')
                return False

    elif isinstance(f,unicode):
        desire_existence(f.encode('ascii','ignore'))
    else:
        raise UserWarning("Unknown type to check existence: '{}'",format(type(f)))
    

def load_ascii_as_array(f):
    if isinstance(f,file):
        array = [x.strip() for x in f.readlines()]
    elif isinstance(f,str):
        require_existence(f)
        array = [x.strip() for x in open(f,'r').readlines()]
    else:
        e ='Unknown type "{}" to load ascii array'.format(type(f))
        cpm.p(e,'r')
        raise TypeError(e)
    return array


def load_ascii_as_matrix(f):
    if isinstance(f,file):
        array = [x.strip().split() for x in f.readlines()]
    elif isinstance(f,str):
        require_existence(f)
        array = [x.strip().split() for x in open(f,'r').readlines()]
    else:
        e ='Unknown type "{}" to load ascii array'.format(type(f))
        cpm.p(e,'r')
        raise TypeError(e)
    return array


try:
    from PySide.QtGui import *
    def qt_choose_existing_file():
        d = QFileDialog()
        d.setFileMode(QFileDialog.ExistingFiles)
        d.setViewMode(QFileDialog.Detail)
        filename = d.getOpenFileName()[0]
        require_existence(filename)
        return filename
    
    def qt_text_input_dialog(parent,title,message):
        text = QInputDialog.getText(parent,title,message)
        print text
        return text[0]

    def qt_yesno_message_box(message,default_no=True):
        box = QMessageBox()
        message = type_util.ensure_string(message)
        box.setText(message)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        if(default_no):
            box.setDefaultButton(QMessageBox.No)
        else:
            box.setDefaultButton(QMessageBox.Yes)
        ret = box.exec_()
        if ret == QMessageBox.Yes:
            return True
        elif ret == QMessageBox.No:
            return False
        else:
            raise RuntimeError("Invalid selection")
except:
    pass
    
