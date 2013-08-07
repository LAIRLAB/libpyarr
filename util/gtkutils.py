#! /usr/bin/env python2.7

import cairo, gtk, gtk.gdk, os, sys, types, gobject, pango
import numpy, math, matplotlib.pyplot as plt
from numpy import pi, sqrt, float32
from math import atan2
from matplotlib.backends.backend_gtkcairo import FigureCanvasGTKCairo as FigureCanvas

from pdbwrap import *
from vis_util import *
import common.util.color_printer as cpm

font_str = "Tahoma 8.4"

class cairo_drawingarea(gtk.DrawingArea):
    def __init__(self):
        super(gtk.DrawingArea, self).__init__()

        self.draw = pdbwrap(self.draw)
        self.offset = numpy.array([0,0], dtype=numpy.int32)
        
        # Draw in response to an expose-event
    __gsignals__ = { "expose-event": "override" }

    def changed(self, what=None):
        self.queue_draw()

    def do_expose_event(self,  event):

        # Create the cairo context
        cairoContext = self.window.cairo_create()

        # Restrict Cairo to the exposed area; avoid extra work
        cairoContext.rectangle(event.area.x, event.area.y,
                     event.area.width, event.area.height)
        cairoContext.clip()

        self.draw(cairoContext, *self.window.get_size())

    def draw(self, cc, w, h):
        pass




class cairo_zoomable_mixin(gtk.DrawingArea):
    def __init__(self):
        self.add_events(gtk.gdk.SCROLL_MASK | gtk.gdk.BUTTON_PRESS_MASK |
                        gtk.gdk.POINTER_MOTION_MASK | gtk.gdk.BUTTON_RELEASE_MASK)
        self.connect('scroll_event', pdbwrap(self.on_scroll))
        self.connect('button_press_event', pdbwrap(self.zoomable_on_press))
        self.connect('button_release_event', pdbwrap(self.zoomable_on_release))
        self.connect('motion_notify_event', pdbwrap(self.zoomable_on_motion))

        self.panning = False
        self.prev = None
        self.offset = numpy.int32([0,0])

        self.scale = 1.0

    def on_scroll(self, widget, event):
        if event.direction == gtk.gdk.SCROLL_UP:
            self.scale *= 0.9
        else:
            self.scale /= 0.9
            
        print self.scale
        self.changed()

    def zoomable_on_motion(self, widget, event):
        if self.panning:
            a = numpy.int32([event.x, event.y])
            self.offset += (a-self.prev)
            self.prev = a
            print self.offset
            self.queue_draw()
            
    def zoomable_on_press(self, widget, event):
        if event.button == 2:
            self.panning = True
            self.prev = numpy.int32([event.x, event.y])
                                    
    def zoomable_on_release(self, widget, event):
        if event.button == 2:
            self.panning = False
            self.prev = None

class cairo_plotwidget(cairo_drawingarea, cairo_zoomable_mixin):
    def __init__(self, parent, xattrs, yattrs):
        cairo_drawingarea.__init__(self)
        cairo_zoomable_mixin.__init__(self)

        self.mparent = parent
        self.mparent.add_dependent(self.changed)
        
        self.xattrs = xattrs
        self.yattrs = yattrs

        self.set_size_request(1280, 720)

    def draw(self, cc, w, h):
        cc.translate(self.offset[0], self.offset[1])
        #cc.scale(self.scale, self.scale)

        print w,h
        cc.scale(w,-h)
        cc.translate(0, -1.0)

        cc.set_source_rgb(1.0,1.0,1.0)
        cc.rectangle(0.0,0.0,1.0,1.0)
        cc.fill()

        cc.set_source_rgb(0,0,0)
        cc.set_line_width(1.0/w)

        for (xattr, yattr) in zip(self.xattrs, self.yattrs):
            this_x = getattr(self.mparent, xattr)
            this_y = getattr(self.mparent, yattr)
            
            cc.move_to(this_x[0], this_y[0])
            print "loop len %d"%len(this_x)
            for i in xrange(len(this_x)):
                if i%10000 == 0:
                    cc.line_to(this_x[i], this_y[i])
            print "done with loop"
            cc.stroke()

class box_n_overlay_widget(cairo_drawingarea, cairo_zoomable_mixin):
    def __init__(self, parent, boxlist_attr=None, 
                 pix_attr='pixarr'):
        cairo_drawingarea.__init__(self)
        cairo_zoomable_mixin.__init__(self)

        self.mparent = parent
        self.mparent.add_dependent(self.changed)
        self.pix_attr = pix_attr
        self.boxlist_attr = boxlist_attr

        self.show_gt = False
        self.show_gt_bboxes = False
        
        self.show_inf = False
        self.show_inf_bboxes = False

        self.show_ta2_bboxes = True
        self.show_ta2_confs = False

        self.min_true_thresh = 0.0

        (self.w, self.h) = getattr(self.mparent, self.pix_attr).shape[:2][::-1]

#        self.set_size_request(500, 500)
        self.set_size_request(self.w, self.h)

    def draw(self, cc, w, h):
        cc.scale(self.scale, self.scale)
        cc.translate(self.offset[0], self.offset[1])

        
        if self.show_gt and self.mparent.gt_arr is not None:
            showarr = float32(getattr(self.mparent, self.pix_attr).copy())
            
            showarr[:,:,1] += 0.4*self.mparent.gt_arr
            showarr[where(showarr > 255)] = 255
            pbuf = make_pixbuf(uint8(showarr))

        elif self.show_inf and self.mparent.inf_arr is not None:
            # convert to float to prevent overflow
            showarr = float32(getattr(self.mparent, self.pix_attr).copy())


            showarr[:,:,0] += 0.4*self.mparent.inf_arr
            showarr[where(showarr > 255)] = 255
            pbuf = make_pixbuf(uint8(showarr))
        elif hasattr(self.mparent, 'heatmap') and self.mparent.use_heatmap:
            pbuf = make_pixbuf(self.mparent.heatmap)
        else:
            pbuf = make_pixbuf(getattr(self.mparent, self.pix_attr))
            
        cc.set_source_pixbuf(pbuf, 0,0)
        cc.paint()

        def show_extra_bboxes(arr, r, g, b):
            if arr is None: return
            
            labelarr, n_labels = label(arr)

            cc.set_source_rgba(r, g, b, 1.0)
            for i in range(n_labels):
                inds = where(labelarr == i+1)
                
                cc.rectangle(inds[1].min(), inds[0].min(),
                             inds[1].max()-inds[1].min(),
                             inds[0].max()-inds[0].min())
                cc.set_line_width(2.0)
                cc.stroke()

        if self.show_gt_bboxes: 
            show_extra_bboxes(self.mparent.gt_arr, 0.0, 1.0, 0.0)

        if self.show_inf_bboxes:
            show_extra_bboxes(self.mparent.inf_arr, 1.0, 0.0, 0.0)

        if self.boxlist_attr is None:
            return

        for (name, boxstruct) in getattr(self.mparent, self.boxlist_attr).iteritems():
            if boxstruct.show:
                for box in boxstruct.boxes:
                    true_conf = box.score

                    if(true_conf < self.min_true_thresh):
                        continue
                        
                    cc.set_source_rgb(*boxstruct.rgb)
                    cc.rectangle(box.x, box.y, 
                                 box.width, box.height)
                    cc.stroke()

                    if (hasattr(box, 'show_conf') and 
                        box.show_conf):

                        alpha = 0.8*true_conf + 0.1

                        cc.set_source_rgba(boxstruct.rgb[0],
                                           boxstruct.rgb[1],
                                           boxstruct.rgb[2],
                                           alpha)
                        cc.rectangle(box.x, box.y,
                                     box.width, box.height)
                        cc.fill()

    def changed(self, what=None):
        self.queue_draw()

class draggable_overlay(box_n_overlay_widget):
    def __init__(self, parent, box_attr=None, pix_attr='pixarr'):
        super(draggable_overlay, self).__init__(parent, box_attr, pix_attr)

        self.cur_dragstart = None
        self.cur_box = None
        self.prev_box = None

        self.connect("button_press_event", pdbwrap(self.on_press))
        self.connect("button-release-event", pdbwrap(self.on_release))
        self.connect("motion-notify-event", pdbwrap(self.on_motion))

        self.add_events(gtk.gdk.POINTER_MOTION_MASK | gtk.gdk.BUTTON_PRESS_MASK  | gtk.gdk.BUTTON_RELEASE_MASK)

    def draw(self, cc, w, h):
        super(draggable_overlay, self).draw(cc, w, h)

        if self.cur_box is not None:
            cc.set_source_rgb(0,0,0)
            cc.rectangle(*self.cur_box)
            cc.stroke()

            cc.set_source_rgb(1,1,1)
            cc.set_dash([1])
            cc.rectangle(*self.cur_box)
            cc.stroke()

    def on_press(self, widget, event):
        if event.button in [1,3]:
            self.cur_dragstart = (event.x, event.y)
            self.prev_box = self.cur_box

    def on_motion(self, widget, event):
        if self.cur_dragstart is not None:
            self.cur_box = (self.cur_dragstart[0],
                            self.cur_dragstart[1],
                            event.x - self.cur_dragstart[0], 
                            event.y - self.cur_dragstart[1])
        self.queue_draw()

    def on_release(self, widget, event):
        self.cur_dragstart = None
        


def draw_histogram(cc, vec, height, norm=None, colors = [], text = False):
    cc.set_line_width(0.005)        
    cc.set_source_rgb(0,0,0)
    cc.rectangle(-1,-1,2,2)
    cc.stroke()
    
    vec_max = height
    if len(colors) != len(vec):
        colors = [(1, 1, .4) for v in vec]

    for (i,v) in enumerate(vec):
        cc.rectangle(i*2.0/len(vec)-1, -1,
                     2.0/len(vec),
                     v*2.0/vec_max)

        cc.set_source_rgb(*colors[i])
        cc.fill()

        cc.rectangle(i*2.0/len(vec)-1, -1,
                     2.0/len(vec),
                     v*2.0/vec_max)


        cc.set_source_rgb(0,0,0)
        cc.stroke()

        #if text:
        #    cc.show_text("score: {}".format(v))


class hist_widget(cairo_drawingarea):
    def __init__(self, mparent, attr, norm):
        cairo_drawingarea.__init__(self)
        
        self.mparent = mparent
        self.mparent.add_dependent(self.changed)
        
        self.set_size_request(4*self.mparent.widget_size_ish/3, 
                              3*self.mparent.widget_size_ish/3)

        self.attr = attr
        self.norm = norm

        self.scale = 1.0


    def draw(self, cc, w, h):
        if getattr(self.mparent, self.attr) is not None:
            cc.scale(self.scale*w/2.0, -self.scale*h/2.0)
            cc.translate(1.0, -1.0)
            cc.rotate(-numpy.pi/2)

            draw_histogram(cc, getattr(self.mparent, self.attr), self.norm)

class bargraph(cairo_drawingarea):
    def __init__(self, mparent, attr, norm, size = (100, 150), colors = []):
        cairo_drawingarea.__init__(self)
        
        self.mparent = mparent
        self.mparent.add_dependent(self.changed)
        
        self.set_size_request(*size)

        self.attr = attr
        self.norm = norm

        self.scale = 1.0
        self.colors = colors

    def draw(self, cc, w, h):
        if getattr(self.mparent, self.attr) is not None:
            cc.scale(self.scale*w/2.0, -self.scale*h/2.0)
            cc.translate(1.0, -1.0)
            #cc.rotate(-numpy.pi/2)

            draw_histogram(cc, 
                           getattr(self.mparent, self.attr), 
                           self.norm, 
                           colors = self.colors, 
                           text = True)

      

def make_pixbuf(r, g=None, b=None):
    if r.__class__.__name__ == 'Image':
        r = numpy.asarray(r, dtype = numpy.uint8)
    if r is None: 
        return None
    if type(r) == type(""):
        return gtk.gdk.pixbuf_new_from_file(r)
    elif g is not None:
        
        retval = numpy.zeros((r.shape[0], r.shape[1], 3), dtype=numpy.uint8)
        retval[:,:,0] = r
        retval[:,:,1] = g
        retval[:,:,2] = b
    elif len(r.shape) == 3 and r.shape[2] == 3:
        retval = r
    elif len(r.shape) == 2:
        retval = numpy.zeros((r.shape[0], r.shape[1], 3), dtype=numpy.uint8)
        retval[:,:,0] = r
        retval[:,:,1] = r
        retval[:,:,2] = r
    else:
        s= "Error: Array of wrong shape passed to make_pixbuf."
        print s
        raise Exception(s)
    
    pixbuf = gtk.gdk.pixbuf_new_from_array(retval, gtk.gdk.COLORSPACE_RGB, bits_per_sample=8)
    
    return pixbuf

def getgraypixbuf(pixbuf):
    realim = numpy.float32(pixbuf.get_pixels_array())
    
    realim = realim/255
    rgbim = nrecImage.rgb()
    rgbim.red = realim[:,:,0]
    rgbim.green = realim[:,:,1]
    rgbim.blue = realim[:,:,2]
    
    (rgbim.rows, rgbim.columns) = rgbim.red.shape

    return rgbim.getGrayscaleArray()

class model(object):
    def __init__(self):
        pass
    def add_dependent(self, d):
        if not(hasattr(self, '_dependents')):
            self._dependents = []
        self._dependents.append(d)
    def remove_dependent(self, d):
        self._dependents.remove(d)
    def changed(self, what=None):
        if hasattr(self, '_dependents'):
            for d in self._dependents:
                d(what)

class prop_model(model):
    def __init__(self, val):
        self._it = val
    def setter(self, val):
        self._it = val
        self.changed()
    def getter(self):
        return self._it
    it = property(getter, setter)

def extend_class(cls):
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func
    return decorator

def gtkpack(container, *widgets):
    """Pack many widgets in a container using pack_start. If an element of widgets is a tuple, it will
    be interpreted as arguments to pack_start: (child, expand, fill, padding)"""
    for w in widgets:
        if w is None:
            pass

        elif isinstance(w, types.TupleType):
            container.pack_start(*w, fill=False)
        else:
            container.pack_start(w, fill=False)

def gtkcolor(*vals):
    return gtk.gdk.Color(red=vals[0]*255,
                         green=vals[1]*255,
                         blue=vals[2]*255)

def gtksetcolor(widget, color):
    states = [gtk.STATE_NORMAL, gtk.STATE_PRELIGHT, gtk.STATE_SELECTED,
              gtk.STATE_INSENSITIVE, gtk.STATE_ACTIVE]
    
    style = widget.get_style().copy()
    map = widget.get_colormap()

    for s in states:
        style.base[s] = map.alloc_color(str(color))
                
    widget.set_style(style)

class gtkradiobuttons:
    def __init__(self, names, cb, colors=None, orient='vert'):
        self.cont = (gtk.VBox if orient=='vert' else gtk.HBox)(spacing=2)
        self.cont.modify_font(pango.FontDescription(font_str))
        # GTK sends an event to each button's callback with
        # some data and whether that button has just
        # become active or inactive; we only care when
        # the button has just become active, so filter
        # the events that way

        self.bs = {}
        self.names = names

        def wrapped_cb(widget, data=None):
            if widget.get_active():
                return cb(widget, data)

        self.bs[names[0]] = gtk.RadioButton(None, names[0])
        self.bs[names[0]].get_child().modify_font(pango.FontDescription(font_str))
        if colors:
            gtksetcolor(self.bs[names[0]], colors[0])

        self.bs[names[0]].connect('toggled', wrapped_cb, names[0])
        self.cont.pack_start(self.bs[names[0]])
        self.bs[names[0]].modify_font(pango.FontDescription(font_str))
        self.bs[names[0]].show()
        self.bs[names[0]].set_active(True)

        for (i, name) in enumerate(names[1:]):
            self.bs[name] = gtk.RadioButton(self.bs[names[0]], name)
            if colors:
                gtksetcolor(self.bs[name], colors[i+1])
            self.bs[name].connect('toggled', wrapped_cb, name)
            self.cont.pack_start(self.bs[name])
            self.bs[name].get_child().modify_font(pango.FontDescription(font_str))
            self.bs[name].show()

    def set_active(self, name):
        if type(name) == type('hi'):
            if 0:
                for n in self.bs:
                    if n == name:
                        self.bs[n].set_active(True)
                    else:
                        self.bs[n].set_active(False)
            else:
                self.bs[name].set_active(True)
        elif type(name) == type(1):
            for (i,n) in enumerate(self.names):
                if i == name:
                    self.bs[n].set_active(True)
                else:
                    self.bs[n].set_active(False)

def gtkhb(*widgets):
    ret = gtk.HBox(spacing=2)
    gtkpack(ret, *widgets)
    return ret

def gtkvb(*widgets):
    ret = gtk.VBox(spacing=2)
    gtkpack(ret, *widgets)
    return ret

def gtkscrolledwindow(widget):
    ret = gtk.ScrolledWindow()
    #ret.set_border_width(10)
    ret.set_policy(gtk.POLICY_AUTOMATIC, 
                   gtk.POLICY_AUTOMATIC)
    ret.add_with_viewport(widget)
    widget.show()
    
    return ret


def gtklv(*widget_lists):
    lstore = gtk.ListStore(*[type(i) for i in widget_lists[0][:2]])
    for w in widget_lists:
        lstore.append(w[:2])
            
            
    tv = gtk.TreeView(model=lstore)
    sel = tv.get_selection()
    sel.set_mode(gtk.SELECTION_MULTIPLE)
    def selection_cb(sel, model, path, is_selected, data):
        # oddly, pygtk sets the is_selected argument to True
        # if it just got deselected, and False otherwise.
        # i think it's supposed to be telling you the previous state. 
        widget_lists[path[0]][2](not(is_selected))
        return True
    sel.set_select_function(selection_cb, None, True)
    cols = []
    cells = []
    for w in widget_lists[0][:2]:
        cols.append(gtk.TreeViewColumn('Label'))
        tv.append_column(cols[-1])
        cells.append(gtk.CellRendererText())
        cols[-1].pack_start(cells[-1])
    for i in range(len(widget_lists[0][:2])):
        if type(widget_lists[0][i]) == type('hi'):
            print "hi"
            cols[i].add_attribute(cells[i], 'text', i)
        elif type(widget_lists[0][i]) == gtk.gdk.Color:
            cols[i].add_attribute(cells[i], 'cell-background-gdk', i)
        else:
            print "oh no", type(widget_lists[0][i])
            
    
    return tv

def gtknb(*widgets):
    """Pack a bunch of (name, widget) tuples into a notebook of tabs."""
    ret = gtk.Notebook()
    for w in widgets:
        w[1].show()
        if w is None:
            pass
        elif isinstance(w, types.TupleType):
            ret.append_page(w[1])
            ret.set_tab_label_text(w[1], w[0])
        else:
            ret.append(w)
    ret.set_tab_pos(gtk.POS_TOP)
    return ret

def gtkbutton(name, cb):
    b = gtk.Button(name)
    b.connect("clicked", pdbwrap(cb), "Hi.")
    b.get_child().modify_font(pango.FontDescription(font_str))
    
    return b

def gtkarrow(name, t, cb):
    b = gtk.Button()
    a = gtk.Arrow(t, gtk.SHADOW_IN)
    a.show()
    b.connect('clicked', pdbwrap(cb), 'hi.')
    b.add(a)
    return b

def gtktogglebutton(name, cb):
    b = gtk.ToggleButton(name)
    def wrap_cb(button, *args):
        cb(button.get_active())
    b.connect('clicked', pdbwrap(wrap_cb), 'Hi.')
    return b

def gtkimagebutton(im, cb):
    b = gtk.Button()
    b.add(im)
    b.connect("clicked", pdbwrap(cb), "Hi.")
    
    return b

def gtk_stockimage_button(id, cb):
    image = gtk.Image()
    image.set_from_stock(id, gtk.ICON_SIZE_SMALL_TOOLBAR)
    return gtkimagebutton(image, cb)

def gtknewbutton(cb):
    return gtk_stockimage_button(gtk.STOCK_NEW, cb)
def gtksavebutton(cb):
    return gtk_stockimage_button(gtk.STOCK_SAVE_AS, cb)
def gtkopenbutton(cb):
    return gtk_stockimage_button(gtk.STOCK_OPEN, cb)

def gtkforwardbutton(cb):
    return gtk_stockimage_button(gtk.STOCK_GO_FORWARD, cb)
def gtkupbutton(cb):
    return gtk_stockimage_button(gtk.STOCK_GO_UP, cb)
def gtkbackbutton(cb):
    return gtk_stockimage_button(gtk.STOCK_GO_BACK, cb)
def gtkdownbutton(cb):
    return gtk_stockimage_button(gtk.STOCK_GO_DOWN, cb)

def gtkundobutton(cb):
    return gtk_stockimage_button(gtk.STOCK_UNDO, cb)
def gtkredobutton(cb):
    return gtk_stockimage_button(gtk.STOCK_REDO, cb)

def gtkplaybutton(cb):
    return gtk_stockimage_button(gtk.STOCK_MEDIA_PLAY, cb)
def gtkpausebutton(cb):
    return gtk_stockimage_button(gtk.STOCK_MEDIA_PAUSE, cb)
def gtkrewindbutton(cb):
    return gtk_stockimage_button(gtk.STOCK_MEDIA_REWIND, cb)

# an itemspec is a tuple like ("Exit", self.exit_callback)
def gtkmenu(name, *itemspecs):
    ret = gtk.Menu()
    ret.set_title(name)
    for (name, callback) in itemspecs:
        it = gtk.MenuItem(name)
        it.connect("activate", callback)
        ret.append(it)
    return ret

def gtkmenubar(*menus):
    ret = gtk.MenuBar()
    for m in menus:
        it = gtk.MenuItem(m.get_title())
        it.set_submenu(m)
        ret.append(it)
    return ret

def gtkentry(cb, default_text=""):
    e = gtk.Entry()
    e.set_text(default_text)
    
    def wrapped_cb(entry):
        return cb(entry.get_text())

    e.connect("activate", wrapped_cb)

    return e

def textdialog(title, okcallback):
    win = gtk.Window()
    win.set_title(title)
    win.set_position(gtk.WIN_POS_CENTER)
    
    def ok_buttoncb(*args):
        okcallback(entry.get_text())
        win.destroy()
    
    def cancel_cb(*args):
        win.destroy()

    entry = gtkentry(ok_buttoncb)
    
    container = gtkvb(entry,
                      gtkhb(gtkbutton("Cancel", 
                                      cancel_cb),
                            gtkbutton("OK",
                                      ok_buttoncb)))
    container.show_all()
    win.add(container)
    win.present()
    return win

def gtklabel(text):
    l = gtk.Label()
    l.set_label(text)
    p = pango.FontDescription(font_str)
    l.modify_font(p)
    return l

def gtkscrollbar(scroll_cb, 
                 lower=0.0,
                 upper = 1.0, 
                 step_incr = .001,
                 page_incr = .01,
                 page_size = .01):

    def wrapped_cb(range, *args):
        return scroll_cb(range.get_adjustment().get_value())
    adj = gtk.Adjustment(value=0.0,
                         lower=lower,
                         upper=upper,
                         step_incr=step_incr,
                         page_incr=page_incr,
                         page_size=page_size)
    slider = gtk.HScrollbar(adj)
    slider.connect("value-changed",pdbwrap(wrapped_cb),"hi.")
    slider.set_update_policy(gtk.UPDATE_CONTINUOUS)
    return slider

def savewindow(save_cb):
    return textdialog("Save", save_cb)

def loadwindow(load_cb):
    return textdialog("Load", load_cb)

def crosshair(cc, x, y, size=0.5):
    cc.set_line_width(size/3)
    cc.move_to(x,size+y)
    cc.line_to(x,-size+y)
    cc.move_to(size+x, y)
    cc.line_to(-size+x, y)
    cc.stroke()

def border(cc, x0=-1, y0=-1, h=2, w=2, color=(0,0,0), line_width=0.01):
    cc.set_source_rgb(*color)
    cc.set_line_width(0.01)
    cc.rectangle(-1,-1,2,2)
    cc.stroke()
    
def scatterplot(cc, samples, color=(1,0,0), radius=0.02):

    for i in range(samples.shape[0]):
        x = samples[i,0]
        y = samples[i,1]

        if len(color)==3:
            cc.set_source_rgb(*color)
        else:
            cc.set_source_rgba(*color)
        cc.arc(x,y,radius,0,2*pi)
        cc.fill()

#axis1 and axis2 are each numpy arrays of floats of shape (2,)
#axis1 and axis2 are assumed to be perpedicular.
def cairo_ellipse(cc, x, y, axis1, axis2):
    mag1 = sqrt(axis1[0]**2 + axis1[1]**2)
    mag2 = sqrt(axis2[0]**2 + axis2[1]**2)
    whichaxis = (axis1 if mag1 > mag2 else axis2)
    angle = math.atan2(whichaxis[0], whichaxis[1])

    cc.save()
    cc.translate(x,y)
    cc.rotate(-angle)

    cc.scale(min(mag1, mag2), max(mag1, mag2))
    cc.arc(0, 0, 1.0, 0, 2*pi)
    cc.restore()


        
    

# Draw an ellipse for the full covariance matrix
# projected onto the i-j plane. 
def cov_ellipse(cc, x, y, covmat, i,j):
    
    curmat = covmat[numpy.array([i,j]),:][:, numpy.array([i,j])]
    
    w,v = numpy.linalg.eig(curmat)

    v = numpy.real(v)

    w = numpy.real(w)

    axis1 = w[0]*v[:,0]
    axis2 = w[1]*v[:,1]

    cairo_ellipse(cc, x,y, axis1, axis2)

    cc.fill()

def choose2from(l1, l2):
    ret = []
    for i in l1:
        for j in l2:
            if (i,j) not in ret and (j,i) not in ret and i != j:
                ret.append((i,j))
    return ret


def draw_arrow(cc, x, y, xdiff, ydiff):

    angle = atan2(xdiff, ydiff)
    len = sqrt(xdiff**2 + ydiff**2)
    cc.save()
    
    cc.translate(x,y)
    cc.rotate(angle)
    
    cc.move_to(0,0)
    cc.line_to(len, 0)
    cc.line_to(0.8*len, 0.2*len)
    cc.move_to(0.8*len, -0.2*len)
    cc.line_to(len, 0)

    cc.restore()


class vis_app(model):
    def __init__(self,
                 image_dir,
                 basenames,    
                 ext):
                 
        self.image_dir = image_dir
        self.basenames = basenames
        self.bn_idx = 0

        self.ext = ext

        self.show_props = []

    def update(self):
        self.basename = self.basenames[self.bn_idx]

        print "BASENAME:",self.basename
        if hasattr(self, 'pic_widget'):
            for a in self.show_props:
                print a,"is",getattr(self.pic_widget, a)

        self.pixarr = numpy.array(Image.open(self.image_dir +
                                             "/" + self.basename + self.ext))
        self.changed()

    def next_cb(self, *args):  
        self.bn_idx += 1
        self.update()

    def prev_cb(self, *args):
        self.bn_idx -= 1
        self.update()

    def filter_cb(self, v):
        self.pic_widget.min_true_thresh = v
        self.changed()

    def def_prop_toggle_cb(self, prop):
        def my_prop_toggle_cb(is_active):
            setattr(self.pic_widget, prop, is_active)
            self.changed()
        return my_prop_toggle_cb

    def def_algcb(self, alg):
        def my_algcb(*args):
            self.algorithm = alg
            self.update()
        return my_algcb
    def make_container(self):
        if hasattr(self, 'window'):
            self.keepalive = True
            self.window.destroy()
            self.keepalive = False

        self.window = gtk.Window()

    def container_show(self):
        self.container.show()
        self.container.show_all()
        self.window.add(self.container)
        self.window.present()
        self.window.move(0,0)

class imshow(draggable_overlay, cairo_zoomable_mixin):
    def __init__(self, parent, attr, click_cb = None):
        cairo_drawingarea.__init__(self)
        cairo_zoomable_mixin.__init__(self)
        super(imshow, self).__init__(parent, pix_attr = attr)
        self.attr = attr
        self.mparent = parent
        self.mparent.add_dependent(self.changed)
        self.set_size_request(360, 240)
        self.connect('button_press_event', pdbwrap(self.on_press))
        self.click_cb = None

    def draw(self, cc, w, h):
        cc.translate(self.offset[0], self.offset[1])
        cc.scale(self.scale, self.scale)
        cc.set_source_pixbuf(make_pixbuf(getattr(self.mparent, self.attr)), 0, 0)
        cc.paint()

    def changed(self, what=None):
        self.queue_draw()

    def on_press(self, widget, event):
        location = (int(event.x), int(event.y))
        print event.button
        if self.click_cb is not None:
            self.click_cb(event)
            return
        else:            
            if event.button == 1:
                try:
                    loc = (event.x, event.y)
                    data = getattr(self.mparent, self.attr) 
                    if data.__class__.__name__ == 'Image':
                        val = data.getpixel(loc)
                    else:
                        val = data[event.y][event.x]
                    #assumes uint8
                    if isinstance(val, list) or isinstance(val, tuple):
                        val = [x / 255.0 for x in list(val)]
                    elif isinstance(val, numpy.ndarray):
                        val = val / 255.0
                    else:
                        val /= 255.0

                    cpm.gcp.info("\n\tclick location: {}\n\tval:{}".format(loc,
                                                                       val))
                except:
                    cpm.gcp.error("imshow::on_press oops")
                    pdb.set_trace()
            #self.mparent.click_cb(self, location)
            if event.button == 3:
                pass
            #self.mparent.toggle_cb(self, location)
