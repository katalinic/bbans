# bbans
TF BB-ANS.

TensorFlow implementation of [BB-ANS](https://arxiv.org/pdf/1901.04866.pdf) by Townsend et al.

Presently, only the binarised MNIST version is provided. The implementation draws inspiration from the [authors' code](https://github.com/bits-back/bits-back).

Both ELBO and compressed message length are ~0.17 bits per pixel (with perfect reconstruction).

To train the model:

`$ python3 train.py -t`

To evaluate the model:

`$ python3 train.py`

To run compression and decompression on e.g. 100 images:

`$ python3 demo.py -n 100`
