#! /usr/bin/env python
import argparse, os, sys
import gen_boilerplate as gen
import x_test_x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clean', action='store_true')
    parser.add_argument('-j', '--nbr-jobs', default=4)
    parser.add_argument('-f', '--use-floats', action='store_true')
    parser.add_argument('--no-test', action = 'store_true')
    args = parser.parse_args()

    if args.use_floats:
        os.environ['CMAKE_USE_FLOATS'] = "YES"
    else: 
        os.environ['CMAKE_USE_FLOATS'] = "NO"

    orig_dir = os.getcwd()

    
    os.chdir(os.environ['LIBPYARR_ROOT'])
    print "curdir",os.getcwd()
    if not args.clean:
        gen.gen_everything()
        os.system('cmake .; make -j%d'%args.nbr_jobs)
    else:
        os.system('rm -rf CMakeFiles/ CMakeCache.txt Makefile bin lib')

    os.system('cd %s'%orig_dir)

    if not args.no_test and not args.clean:
        x_test_x.test_all_modules()
    elif not args.clean:
        print "\n\nNOT RUNNING LIBPYARR TEST! \n\n"
    

if __name__=='__main__':
    gen.pdbwrap(main)()
