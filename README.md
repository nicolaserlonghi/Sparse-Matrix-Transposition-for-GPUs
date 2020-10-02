# Sparse-Matrix-Transposition-for-GPUs

## Usage

The build process tool of the project is Makefile. The following are the steps to successfully build and run the project:

1. Clone or download this repository and move to the folder project.

2. If your device have python3 installed do this step and you have done otherwise go direct to the step three.

```
python3 run.py [folder that contains matrixs.mtx][single matrix.mtx]
```

3. if you got here it means you don't have python3 so let's do it by hand.

```
make
make install
```

4. Run it on a single matrix.

```
./bin/sparse_matrix_transpose matrix.mtx
```

## Test files

Some test files can be found in the [testFiles folder](./testFiles/). If you want to download others matrixs you can find them [here](https://sparse.tamu.edu) and to get them in the right format you have to click on *Matrix Market* button.
