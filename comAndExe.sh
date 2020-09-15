rm -rf ./bin
make
make install
make clean
./bin/sparse_matrix_transpose testFiles/myMatrix.mtx