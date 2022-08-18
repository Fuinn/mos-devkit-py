find . -name \*.so -delete
find . -name \*.pyc -delete
find . -name \*~ -delete
find . -name __pycache__ -delete
find . -name _dmumps.c -delete
find . -name _cbc.c -delete
find . -name _clp.c -delete
find . -name _ipopt.c -delete
find . -name _eval.c -delete
rm -rf *.egg-info
rm -rf build
rm -rf dist
