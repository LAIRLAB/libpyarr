import os

def test_all_modules():
    cmd = 'py.test -v --maxfail 1 -v --doctest-glob nomatch'


    orig_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.system(cmd)
    os.chdir(orig_dir)
