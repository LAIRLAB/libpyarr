#! /usr/bin/env python
import os, sys, argparse, random
import copy

from common.util.pdbwrap import *
import common.util.color_printer as cpm
import common.util.file_util as fu
import common.util.img_util as iu

class FoldMaker(object):
    def __init__(self, image_dir, gt_dir, split_ratio, number_of_folds = 3, unsup_ratio = None):
        i_names = os.listdir(image_dir)
        gt_names = os.listdir(gt_dir)
        i_bnames = [x.split('.')[0] for x in i_names]
        gt_bnames = [x.split('.')[0] for x in gt_names]

        #uniquify and sort
        basenames = sorted(list(set([x for x in gt_bnames if x in i_bnames])))

        cpm.gcp.info("{} shared basenames".format(len(basenames), image_dir))
        num_bad = fu.verify_basenames(basenames, image_dir, gt_dir)
        cpm.gcp.warning("{} bad image - gt pairs".format(num_bad))
        cpm.gcp.info("{} valid image - gt pairs will be used for folds".format(len(basenames)))

        num_train = int(split_ratio * len(basenames))
        num_test = len(basenames) - num_train
        if unsup_ratio is not None:
            num_unsup = int(unsup_ratio * len(basenames))
        
        self.data = {}
        self.data['all'] = copy.deepcopy(basenames)

        random.shuffle(basenames)

        for f in range(number_of_folds):
            self.data[(f, 'train')] = []
            self.data[(f, 'test')] = []
            if unsup_ratio is not None:
                self.data[(f, 'uns_train')] =[]
    
            for (c, b) in enumerate(basenames):
                if c < num_train:
                    self.data[(f, 'train')].append(b)
                    self.data[(f, 'uns_train')].append(b)                    
                elif unsup_ratio is not None and c < num_unsup:
                    self.data[(f, 'uns_train')].append(b)                    
                else:
                    self.data[(f, 'test')].append(b)
                    
            random.shuffle(basenames)

        ne = 5
        num_toy = min(ne, min(num_train, num_test))
        self.data[(0, 'traintoy')] = []
        self.data[(0, 'testtoy')] = []
        for n in range(num_toy):
            self.data[(0, 'traintoy')].append(basenames[n])
            self.data[(0, 'testtoy')].append(basenames[n + ne])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image-dir', required = True)
    parser.add_argument('-g', '--gt-dir', required = True)
    parser.add_argument('-o', '--output-dir', required = True)
    parser.add_argument('-n', '--number-of-folds', type = int, default = 3)
    parser.add_argument('-s', '--split-ratio', type = float, default = 0.8)
    parser.add_argument('-v', '--verbosity', default = 'info')
    parser.add_argument('--unsupervised-split', action = 'store_true')
    parser.add_argument('--unsupervised-ratio', type = float, default = .8)

    args = parser.parse_args()

    cpm.gcp.verbosity = args.verbosity

    unsup_ratio = None if not args.unsupervised_split else args.unsupervised_ratio
    if unsup_ratio is not None:
        assert(args.split_ratio <= args.unsupervised_ratio)

    fm = FoldMaker(args.image_dir, 
                   args.gt_dir, 
                   args.split_ratio, 
                   args.number_of_folds, 
                   unsup_ratio = unsup_ratio )

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    with open('{}/all.txt'.format(args.output_dir), 'w') as f:
        f.write('\n'.join(fm.data['all']))
    with open('{}/fold_traintoy_0.txt'.format(args.output_dir), 'w') as f:
        f.write('\n'.join(fm.data[0, 'traintoy']))
    with open('{}/fold_testtoy_0.txt'.format(args.output_dir), 'w') as f:
        f.write('\n'.join(fm.data[0, 'testtoy']))
        
    uns_string = '' if not args.unsupervised_split else 'uns'
    sup_string = '' if not args.unsupervised_split else 'sup'

    if not args.unsupervised_split:
        for i in range(args.number_of_folds):
            with open('{}/fold_train_{}_{}.txt'.format(args.output_dir,                                    
                                                       i,
                                                       int(100 * args.split_ratio)), 'w') as f:
                f.write('\n'.join(fm.data[(i, 'train')]))
                
            with open('{}/fold_test_{}_{}.txt'.format(args.output_dir, 
                                                      i,
                                                      100 - int(100 * args.split_ratio)), 'w') as f2:
                f2.write('\n'.join(fm.data[(i, 'test')]))
    else:
        sup_per = int(100 * args.split_ratio)
        unsup_per = int(100 * args.unsupervised_ratio)
        test_per = 100 - unsup_per

        for i in range(args.number_of_folds):            
            with open('{}/fold_trainSup_{}_{}_{}_{}.txt'.format(args.output_dir,
                                                                i,
                                                                sup_per,
                                                                unsup_per,
                                                                test_per), 'w') as f:
                f.write('\n'.join(fm.data[(i, 'train')]))            

            with open('{}/fold_trainUns_{}_{}_{}_{}.txt'.format(args.output_dir,
                                                                i,
                                                                sup_per,
                                                                unsup_per,
                                                                test_per), 'w') as f:
                f.write('\n'.join(fm.data[(i, 'uns_train')]))

            with open('{}/fold_test_{}_{}_{}_{}.txt'.format(args.output_dir, 
                                                            i,
                                                            sup_per,
                                                            unsup_per,
                                                            test_per), 'w') as f:
                f.write('\n'.join(fm.data[(i, 'test')]))



if __name__ == '__main__':
    pdbwrap(main)()
