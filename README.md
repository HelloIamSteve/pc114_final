# pc114_final
A C++ / CUDA implementation of LeNet inference on MNIST. This project compares the inference speed of:
1. Sequential C++
2. Pthread
3. OpenMP
4. CUDA

The LeNet model is trained in PyTorch first.<br>
The trained weights and MNIST testing set are then exported into text/image files so that the C++ program can load them directly.

---

## Environment
This project requires PyTorch to export the weights and dataset (and retrain/test the model).
To export pre-trained weights in the project, use:
```bash
cd ./PyTorch
python3 ./export_weights.py
```

And export the dataset:
```bash
cd ./PyTorch
python3 ./export_testing_set.py
```

---

## Build
To compile the project:
```bash
make clean
make
```

## Execute
```bash
./main <THREAD_NUM> <BLOCK_SIZE> <TEST_NUM>
```

Argument meaning:

| Argument | Meaning |
|---|---|
| `THREAD_NUM` | Number of CPU threads used by Pthread and OpenMP versions |
| `BLOCK_SIZE` | Number of CUDA threads per block |
| `TEST_NUM` | Number of test executions|

e.g. 
```bash
./main 6 32 5
```
Will use 6 threads for the pthreads and openMP versions, and 32 threads for each CUDA block, test 5 times and calculate the average execution time.