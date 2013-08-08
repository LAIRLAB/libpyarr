#! /usr/bin/env python

'''
Recursively looks for EPD
Writes to STDOUT and STDERR the found library and the found include directory. In this way,
this script can be executed within CMAKE and the Python Libraries and Includes can be set to the STDOUT and STDERR streams

Checks for a minimum version of Python, default 2.7, but can be specified as a command-line argument

The root paths it starts looking for EPD in are contained in the 'check_dirs' global variable
'''

import os, sys, getpass

check_dirs = ['/home/%s' % getpass.getuser(), '/usr/share/', '/home/ecuzzill', '/opt/local', '/opt']

def main():
    if len(sys.argv) == 2:
        min_py_version = sys.argv[1]

        #make sure it's a version by casting to float
        try:
            min_py_version = 'libpython' + str(float(min_py_version))
        except ValueError:
            min_py_version = 'libpython2.7'
    else:
        min_py_version = 'libpython2.7'

    found = False
    for d in check_dirs:
        for (dname, dnames, fnames) in os.walk(d):
            for r in dnames:

                #found an 'epd'-ish directory
                if r.find('epd') >= 0:
                    full_dir = '%s/%s' % (dname, r)
                    lib_exists = False

                    #find the library
                    for lib in [x for x in os.listdir(full_dir + '/lib/') if x.find('libpython') >=0 ]:
                        lib_version = '.'.join(lib.split('.')[:2])
                        if lib_version >= min_py_version:
                            lib_exists = True
                            break
                    if not lib_exists:
                        break
                    
                    lib = '%s/lib/%s.so' % (full_dir, lib_version)
                    include_dir = '%s/include/%s' % (full_dir, min_py_version[3:])

                    #success if this passes
                    if os.path.isfile(lib) and os.path.isdir(include_dir):
                        sys.stdout.write(lib)
                        sys.stderr.write(include_dir)
                        found = True
            break
        if found:
            break

if __name__ == '__main__':
    main()
