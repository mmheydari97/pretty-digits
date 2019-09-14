# Pretty Digits
In this project, we compare machine learning algorithms to predict Farsi hand-written digits.
The simplest way is to employ a K-Nearest Neighbors model. However, a Deep Neural Network with fully connected layers increases accuracy.
Finally, fitting a Convolutional Neural Network reaches us to 99 percent accuracy after a few epochs.

## Hoda dataset
It is just like MNIST dataset, but with Persian digits. the samples are all hand-written, grayscale, at the center and without rotation.
That is why I called this repository "pretty-digits".
I also used [HodaDatasetReader](https://github.com/amir-saniyan/HodaDatasetReader) to get .cdb files and read them.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
