#!/usr/bin/env bash

rm -rf data
rm test_result.txt
python setup.py sdist
pip uninstall vivid -y &&\
 pip install $(ls dist/*.tar.gz)[test]

[ $# -ge 1 ] && dir=$1 || dir=./tests/
pytest ${dir}
