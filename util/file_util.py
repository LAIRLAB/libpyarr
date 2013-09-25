import color_printer as cpm
import type_util as tu
import os, sys, type_util, datetime, Image, numpy, img_util, cPickle, pdb

rm_rf_fixed = False

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
    arrfiles = [dirname + '/arr_%d.npz'%i for i in xrange(len(files))]
    
    if len(arrfiles) != len(files):
        print "oh no there's crap in a savearr dir:", dirname

    return numpy.concatenate([numpy.load(f)['arr_0'] for f in arrfiles], 
                             axis=0)

def load_pkl(fname, mod_class_dict = {}):
    if isinstance(mod_class_dict, dict):
        for (m, classes) in mod_class_dict.items():
            mod = __import__(m, fromlist=classes)
            tu.add_classes_to_main(mod.__name__, classes)
    elif isinstance(mod_class_dict, tuple):
        m, classes = mod_class_dict[0], mod_class_dict[1]
        mod = __import__(m, fromlist=classes)
        tu.add_classes_to_main(mod.__name__, classes)
        
    f = open(fname)
    ret = cPickle.load(f)
    f.close()
    return ret

def save_pkl(obj, fname):
    f = open(fname, 'w')
    cPickle.dump(obj, f, cPickle.HIGHEST_PROTOCOL)
    f.close()

def path_relative_to_file(file_path, path):
    return os.path.abspath(os.path.dirname(file_path) + '/' + path)

def basename_split(bname):
    return bname.strip('/').split('/')[-1].split('.')[0]

def gts(m = True, s = False, ms = False):
    return generate_timestamp(m, s, ms)

def save_vector_hierarchy(vh):
    h = []
    for i in range(len(vh)):
        rl = vh.at(i)
        

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

def load_basenames(basenames_fn):
    return [b.strip() for b in open(basenames_fn).readlines()]
   
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
    
class DiskCache(object):
    def __init__(self, d, loadable = True):
        d = os.path.abspath(d)
        if not os.path.isdir(d):
            os.mkdir(d)
        self.base = d
        self.loadable = loadable

    def __repr__(self):
        s = "DiskCache\n"
        s += "\tBase dir: {}\n".format(self.base)
        s += "\tLoadable: {}".format(self.loadable)
        return s

    def load(self, b, identifier, rf, ext=''):
        if not self.loadable:
            cpm.gcp.warning("Not loading from cache")
            return None
        fn = self.base + '/{}_{}{}'.format(b, identifier, ext)
        try:
            x = rf(fn)
            cpm.gcp.debug("Cache hit: {}".format(fn))
            return x
        except IOError:
            cpm.gcp.msg("Cache miss: {}".format(fn))
            return None

    def save(self, obj, b, identifier, sf, ext = '', overwrite=False, reverse_sf = False):
        fn = self.base + '/{}_{}{}'.format(b, identifier, ext)
        if os.path.isfile(fn):
            if not overwrite:
                raise RuntimeError(\
                    cpm.gcp.error("Not overwriting cached file!"))
            else:
                cpm.gcp.warning("Overwriting cached file!")

        if reverse_sf:
            sf(obj, fn)
        else:
            sf(fn, obj)
        return True

#snippet for tracking open files        
"""import __builtin__
openfiles = set()
oldfile = __builtin__.file
class newfile(oldfile):
    def __init__(self, *args):
        self.x = args[0]
        print "### OPENING %s ###" % str(self.x)            
        oldfile.__init__(self, *args)
        openfiles.add(self)

    def close(self):
        print "### CLOSING %s ###" % str(self.x)
        oldfile.close(self)
        openfiles.remove(self)
oldopen = __builtin__.open
def newopen(*args):
    return newfile(*args)
__builtin__.file = newfile
__builtin__.open = newopen"""
