# pca-ann-facial-recognition
Facial recognition implemented in python employing principal component analysis and artificial neural networks (PCA-ANN).

### Quick summary of PCA-ANN
The dataset is first split into testing and training faces.

Eigenfaces are generated from the training dataset using principal component analysis, and essentially represent standardized face ingredients. A set of eigenvalues can be extracted from a face using the eigenfaces, which represent how much the face matches each given eigenface. If you have 10 generated eigenfaces, then the 10 eigenvalues can be combined to recreate to reconstruct the original face.

A neural network is created for each of the subjects being recognized (15 subjects, so 15 neural networks), and trained on the eigenvalues of all the training images. It is tested on the eigenvalues of the faces in the testing dataset, which are extracted from the original eigenfaces generated by the training dataset.

Once the networks are trained, a face is recognised by first generating its eigenvalues, then these values are run through each of the 15 subject networks. The network with the highest confidence of a match is the chosen one.

### Usage
Run `main.py` to use an interactive terminal program where you can create, train, test, evaluate, and view a PCA-ANN network with configurable parameters.


## Screenshots

![test_faces](./screenshots/test_faces.png)

![eigenfaces](./screenshots/eigenfaces.png)

![reconstructed_faces](./screenshots/reconstructed_faces.png)

Above are faces generated by combining the eigenfaces in various ways. Compare these to the original test faces.

![accuracy_trend](./screenshots/accuracy_trend.png)
