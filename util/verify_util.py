import os, sys, numpy

import color_printer as cpm

#check dimensions and that each row sums to 1.
def verify_2d_distribution(array, num_segs, num_labels, tolerance = 0.000001):
    desired_shape = (num_segs, num_labels)
    assert array.shape == desired_shape, "distribution shape is not {}".format(desired_shape)

    mass_sum = numpy.sum(array, 1)
    mass_info = "rows are not probability mass functions"
    assert numpy.sum(mass_sum > 1 - tolerance) == num_segs, mass_info
    assert numpy.sum(mass_sum < 1 + tolerance) == num_segs, mass_info

def verify_3d_distribution(array, height, width, labels, tolerance = 0.000001):
    desired_shape = (height, width, labels)
    assert array.shape == desired_shape, "array shape is not {}".format(desired_shape)
    num_pixels = width*height
    mass_sum = numpy.sum(array, 2)
    mass_info = "3d dimension is not a probability mass function"

    assert numpy.sum(mass_sum > 1 - tolerance) == num_pixels, mass_info
    assert numpy.sum(mass_sum < 1 + tolerance) == num_pixels, mass_info
