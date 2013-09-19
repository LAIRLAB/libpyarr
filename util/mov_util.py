import Image, numpy
import libnrec.util.rand_util as ru
import os

def three_d_grayscale_series_to_movie(a, framerate = 5):
    assert(a.ndim == 3)
    length = a.shape[2]
    all_fns = []
    assert(length < 9999)

    tmp_dir = os.getcwd() + '/' + ru.random_ascii_string(5)
    os.mkdir(tmp_dir)

    cmd = 'ffmpeg -r {} -i {}/%04d.png -y {}/{} '.format(framerate,
                                                         tmp_dir,
                                                         tmp_dir,
                                                         'movie.avi')


    for frame_idx_n in range(length):
        frame_n = a[:, :, frame_idx_n]
        fn = '{}/{:04d}.png'.format(tmp_dir, frame_idx_n)
        Image.fromarray(frame_n).save(fn)
        all_fns.append(fn)
    os.system(cmd)

def rgb_series_to_movie(a, framerate = 5):
    assert(a.ndim == 4)
    length = a.shape[3]
    all_fns = []
    assert(length < 9999)

    tmp_dir = os.getcwd() + '/' + ru.random_ascii_string(5)
    os.mkdir(tmp_dir)

    cmd = 'ffmpeg -r {} -i {}/%04d.png -y {}/{} '.format(framerate,
                                                         tmp_dir,
                                                         tmp_dir,
                                                         'movie.avi')


    for frame_idx_n in range(length):
        frame_n = a[:, :, :, frame_idx_n]
        fn = '{}/{:04d}.png'.format(tmp_dir, frame_idx_n)
        Image.fromarray(frame_n).save(fn)
        all_fns.append(fn)
    os.system(cmd)

    
