import numpy as np 
import pandas as pd 
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

# fixes matplotlib not working on linux, haven't tested for windows
import matplotlib
matplotlib.use("TkAgg")

import warnings
warnings.filterwarnings("ignore")


class Subject_ANN:
    # the PCA-ANN needs a neural network for each subject.
    # this class is used for every subject

    def __init__(self, index, train_classes, test_classes):
        self.index = index

        # generate unique classes for subject's ann
        # currently y_train and y_test have a class for each subject,
        # but we want two classes: 1 if it is the subject and 0 if it is not
        self.y_train = self.create_y(train_classes)
        self.y_test = self.create_y(test_classes)
    
    def create_y(self, y):
        # generate training datself.hidden_layersa from classifiers
        arr = []
        for elem in y:
            if elem == self.index:
                arr.append(1.0)
            else:
                arr.append(0.0)
        return pd.Series(arr)

    def create_network(self, **kwargs):
        # multilayer perceptron
        self.neural_network = MLPClassifier(**kwargs)

    def train(self, x_train, x_test, n_epochs, batch_size):
        # custom train method is used so that we can record
        # accuracy scores for each epoch (used for graphing)
        self.train_scores = []
        self.test_scores = []
        n_samples = x_train.shape[0]

        for epoch in range(n_epochs):
            perm = np.random.permutation(n_samples)  # shuffle samples

            for batch_index in range(0, n_samples, batch_size):
                batch = perm[batch_index:batch_index + batch_size]  # create batch
                # train with batch
                self.neural_network.partial_fit(
                    x_train[batch],
                    self.y_train[batch],
                    classes=[0, 1]
                )
            # record scores
            self.train_scores.append(
                self.neural_network.score(x_train, self.y_train)
            )
            self.test_scores.append(
                self.neural_network.score(x_test, self.y_test)
            )

    def show_graph(self):
        # graph of training and test accuracy
        fig, ax = plt.subplots(2, sharex=True, sharey=True)
        ax[0].plot(self.train_scores)
        ax[0].set_title("Train")
        ax[1].plot(self.test_scores)
        ax[1].set_title("Test")
        fig.suptitle(
            "Accuracy over epochs for subject {}".format(self.index + 1),
            fontsize=14
        )
        plt.show()
    
    def test(self, x_test_pca):
        # generate a classification report showing accuracy
        y_pred = self.neural_network.predict(x_test_pca)
        print(classification_report(self.y_test, y_pred))
    
    def predict(self, face_descriptor):
        # predict whether face_descriptor is the
        # subject this network covers or not
        return self.neural_network.predict_proba(face_descriptor)



class PCA_ANN:
    # this class is used to manage the PCA-ANN.

    def __init__(self, n_sub, df, tt_ratio, n_efaces, h_layers, im_size):
        self.image_size = im_size
        self.n_subjects = n_sub

        self.is_trained = False
        self.hidden_layers = h_layers  # hidden layers for subject_anns
        self.n_eigenfaces = n_efaces
        self.train_test_ratio = tt_ratio

        self.n_train = 0
        self.n_test = 0

        self.split_train_test(df)
        self.create_pca()
        self.apply_pca()
        self.create_subject_anns()

    def split_train_test(self, df):
        # desired outputs (train and testing for images and classifications)
        self.train_images = pd.DataFrame()
        self.test_images = pd.DataFrame()
        self.train_classes = []
        self.test_classes = []

        # final column in data file is the classification
        # if we group by the classification and then build the
        # train/test arrays up from there, we can assure there
        # is an equal share of subjects in each.
        for i, x in df.groupby('target'):
            x = shuffle(x)
            split = int(self.train_test_ratio * len(x))
            self.train_images = self.train_images.append(x[:split])
            self.test_images =  self.test_images.append(x[split:])
            self.train_classes += [i for _ in range(split)]
            self.test_classes += [i for _ in range(len(x) - split)]

        # drop the classification from the image data
        self.train_images = self.train_images.drop("target", axis=1)
        self.test_images = self.test_images.drop("target", axis=1)

        # update attributes
        self.n_train = self.train_images.shape[0]
        self.n_test = self.test_images.shape[0]

        # shuffle the data since it's currently sorted by classification
        self.train_images, self.train_classes = shuffle(
            self.train_images, self.train_classes
        )
        self.test_images, self.test_classes = shuffle(
            self.test_images, self.test_classes
        )

    def create_pca(self):
        self.pca = PCA(
            n_components=self.n_eigenfaces, whiten=True
        ).fit(self.train_images)

    def apply_pca(self):
        # create face descriptors from the image data.
        # these face descriptors will be used to train the networks
        self.train_descriptors = self.pca.transform(self.train_images)
        self.test_descriptors = self.pca.transform(self.test_images)

    def create_subject_anns(self):
        # create a neural network for every subject
        self.subject_anns = []
        for i in range(self.n_subjects):
            subject_ann = Subject_ANN(i,
                  self.train_classes, self.test_classes
            )
            subject_ann.create_network(
                hidden_layer_sizes=self.hidden_layers
            )
            self.subject_anns.append(subject_ann)

    def train(self, n_epochs, batch_size):
        # training already completed, reset for new train
        if self.is_trained:
            self.create_subject_anns()

        for index, subject_ann in enumerate(self.subject_anns, 1):
            print("  Training subject_ann {}/{}...".format(
                index, self.n_subjects
            ))
            subject_ann.train(
                self.train_descriptors, self.test_descriptors,
                n_epochs, batch_size
            )

        self.is_trained = True

    def predict(self, face_descriptor):
        # predict face descriptor in every subject_ann to find highest match
        highest = 0
        h_index = 0
        for index, subject_ann in enumerate(self.subject_anns):
            pred = subject_ann.predict(face_descriptor)[0][1]
            if pred > highest:
                highest = pred
                h_index = index

        # return (chosen index, confidence)
        return (h_index, highest)

    def test(self):
        score = 0

        print("")
        print(" Guess | Answer | Confidence ")
        print("-------+--------+------------")

        for i in range(self.n_test):
            face_descriptor = self.test_descriptors[i:i+1]
            choice, confidence = self.predict(face_descriptor)
            if choice == self.test_classes[i]:
                score += 1

            a = str(choice)
            b = str(self.test_classes[i])
            c = str(round(confidence, 3))

            print(" " * (6 - len(a)) + a + " |",
                  " " * (6 - len(b)) + b + " |",
                  " " * (10 - len(c)) + c)
        print("-------+--------+------------")
        print("Score:",
              str((score / self.n_test) * 100) + "%"
        )

    def test_subject_ann(self, i):
        self.subject_anns[i].test(self.test_descriptors)

    def show_graph(self):
        fig, ax = plt.subplots(2, sharex=True, sharey=True)
        ax[0].set_title("Train")
        ax[1].set_title("Test")
        for index, subject_ann in enumerate(self.subject_anns, 1):
            ax[0].plot(subject_ann.train_scores)
            ax[1].plot(subject_ann.test_scores)
            plt.plot(subject_ann.train_scores, label=str(index))
        fig.suptitle("Accuracy over epochs", fontsize=14)
        plt.show()

    def show_eigenfaces(self):
        # show the eigenfaces used to generate face descriptors
        fig, axes = plt.subplots(5, 5, figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            ax.set_yticks([])
            ax.set_xticks([])
            try:
                ax.imshow(
                    self.pca.components_[i].reshape(*self.image_size),
                    cmap='gray'
                )
            except IndexError:
                pass
        fig.suptitle("Eigenfaces", fontsize=14)
        plt.show()

    def show_train_faces(self):
        # show a few train images
        fig, axes = plt.subplots(5, 5, figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            ax.set_yticks([])
            ax.set_xticks([])
            try:
                ax.imshow(
                    np.array(self.train_images)[i].reshape(*self.image_size),
                    cmap="gray"
                )
            except IndexError:
                pass
        fig.suptitle("Train faces", fontsize=14)
        plt.show()

    def show_test_faces(self):
        # show a few test images
        fig, axes = plt.subplots(5, 5, figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            ax.set_yticks([])
            ax.set_xticks([])
            try:
                ax.imshow(
                    np.array(self.test_images)[i].reshape(*self.image_size),
                    cmap="gray"
                )
            except IndexError:
                pass
        fig.suptitle("Test faces", fontsize=14)
        plt.show()

    def show_reconstructed_faces(self):
        # rebuilds train images from face descriptors
        reconstructed = self.pca.inverse_transform(self.train_descriptors)
        fig, axes = plt.subplots(5, 5, figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            ax.set_yticks([])
            ax.set_xticks([])
            try:
                ax.imshow(
                    np.array(reconstructed)[i].reshape(*self.image_size),
                    cmap="gray"
                )
            except IndexError:
                pass
        fig.suptitle("Reconstructed faces", fontsize=14)
        plt.show()

    def show_constructed_faces(self):
        # recreates test images from the  face descriptors gained from training
        # the PCA with test images
        constructed = self.pca.inverse_transform(self.test_descriptors)
        fig, axes = plt.subplots(5, 5, figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            ax.set_yticks([])
            ax.set_xticks([])
            try:
                ax.imshow(
                    np.array(constructed)[i].reshape(*self.image_size),
                    cmap="gray"
                )
            except IndexError:
                pass
        fig.suptitle("Constructed faces", fontsize=14)
        plt.show()
