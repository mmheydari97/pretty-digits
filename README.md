# Pretty Digits
Using machine learning algorithms to predict hand-written digits.
The simplest way is to employ a K-Nearest-Neighbors model. However, a Deep Neural Network increases the accuracy.
Finally, fitting a Convolutional Neural Network reaches us to 99 percent accuracy after a few epochs.

## Hoda dataset
It is just like mnist dataset, but with persian digits. the samples are all hand-written, gray scale, at the center and without rotation.
That is why I called this repository "pretty-digits".
I also used [HodaDatasetReader](https://github.com/amir-saniyan/HodaDatasetReader) to get .cdb files and read them.
