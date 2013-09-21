import Image, numpy
import libnrec.util.rand_util as ru
import libnrec.util.color_printer as cpm
import os


def three_d_grayscale_series_to_movie(a, framerate = 5):
    assert(a.ndim == 3)
    length = a.shape[2]
    all_fns = []
    assert(length < 9999)

    tmp_dir = os.getcwd() + '/' + ru.random_ascii_string(5)
    os.mkdir(tmp_dir)
    cpm.gcp.msg("making movie in {}".format(tmp_dir))

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

def rgb_series_to_movie(a, framerate = 5, final_frame_dup = 10):
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

    counter = frame_idx_n + 1
    cpm.gcp.info("here: {}".format(final_frame_dup))
    for ffd in range(final_frame_dup):
        frame_n = a[:, :, :, frame_idx_n]
        fn = '{}/{:04d}.png'.format(tmp_dir, counter)
        Image.fromarray(frame_n).save(fn)
        all_fns.append(fn)
        counter += 1
    os.system(cmd)

    
