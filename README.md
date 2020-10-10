# Sparse-Matrix-Transposition-for-GPUs

## Usage

The build process tool of the project is Makefile. The following are the steps to successfully build and run the project:

1. Clone or download this repository and move to the folder project.

2. Download test files and unarchive them.

```
wget -i testFilesLink.txt -P testFiles
mv unzip.sh testFiles/
cd testFiles/
chmod +x unzip.sh
./unzip.sh
mv unzip.sh ..
cd ..
```

3. If your device have python3 installed do this step and you have done otherwise go direct to the step four.

```
python3 run.py [folder that contains matrices.mtx][single matrix.mtx]
```

4. if you got here it means you don't have python3 so let's do it by hand.

```
make
make install
```

5. Run it on a single matrix.

```
./bin/sparse_matrix_transpose matrix.mtx
```

## Test files

If you want to download others matrices you can find them [here](https://sparse.tamu.edu) and to get them in the right format you have to click on *Matrix Market* button.
