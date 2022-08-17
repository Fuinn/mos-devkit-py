find . -name \*.so -delete
find . -name \*.pyc -delete
find . -name \*.c -delete
find . -name \*~ -delete
find . -name __pycache__ -delete
find . -name libipopt* -delete
find . -name libcoinmumps* -delete
find . -name libClp* -delete
find . -name libCbc* -delete
rm -rf *.egg-info
rm -rf build
rm -rf dist
rm -rf lib
