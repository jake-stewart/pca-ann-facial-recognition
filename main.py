from cli_utils import input_int, input_float, alert
from pca_ann import PCA_ANN
import pandas as pd
import pickle
import traceback

N_SUBJECTS = 15
IMAGE_SIZE = (64, 64)

def clear_screen():
    print(100 * "\n")


def pca_ann_cli(file_name, df):
    pca_ann = None

    while True:
        # display main menu
        clear_screen()
        print("[1] New PCA-ANN network")
        print("[2] Load existing network")
        print("[3] Save network")
        print("[4] View network")
        print("[5] Train network")
        print("[6] Test network")
        print("[7] Classification reports")
        print("[q] Quit")
        user_input = input("> ").strip().lower()

        # quit
        if user_input == "q":
            exit()

        # new network
        elif user_input == "1":
            n_eigenfaces     = input_int("Number of eigenfaces: ", 60, 1, 120)
            train_test_ratio = input_float("Train test ratio", 0.8, 0, 0.99)
            n_hidden_layers  = input_int("Number of hidden layers", 1, 0)
            hidden_layers    = []
            for i in range(n_hidden_layers):
                prompt = "  Number of nodes for hidden layer {}".format(i + 1)
                nodes = input_int(prompt, 60, 1)
                hidden_layers.append(nodes)

            try:
                pca_ann = PCA_ANN(
                    N_SUBJECTS, df, train_test_ratio,
                    n_eigenfaces, hidden_layers, IMAGE_SIZE
                )

            except Exception as e:
                traceback.print_exc()
                pca_ann = None
                alert("Error: Exception raised creating network.")

        # load network
        elif user_input == "2":
            try:
                pca_ann = pickle.load(open(file_name, "rb"))
                alert("Networked loaded.")
            except FileNotFoundError:
                alert("Error: File not found.")

        # save network
        elif user_input == "3":
            if not pca_ann:
                alert("Error: No network exists to save.")
            else:
                pickle.dump(pca_ann, open(file_name, "wb"))
                alert("Networked saved.")

        # view network
        elif user_input == "4":
            if not pca_ann:
                alert("Error: No network exists to view.")
            else:
                while True:
                    clear_screen()
                    print("Details")
                    print("  Trained:", pca_ann.is_trained)
                    print("  Train/test ratio:", pca_ann.train_test_ratio)
                    print("  Number of train faces:", pca_ann.n_train)
                    print("  Number of test faces:", pca_ann.n_test)
                    print("  Number of eigenfaces:", pca_ann.n_eigenfaces)
                    print("  Hidden layer structure:", pca_ann.hidden_layers)
                    print("")
                    print("Graphs")
                    print("  [1] Accuracy trend of entire PCA-ANN")
                    print("  [2] Accuracy trend of individual subject ANN")
                    print("")
                    print("Faces")
                    print("  [3] Eigenfaces")
                    print("  [4] Train faces")
                    print("  [5] Train faces reconstructed by eigenfaces")
                    print("  [6] Test faces")
                    print("  [7] Test faces constructed by eigenfaces")
                    print("")
                    print("[q] Main menu")
                    user_input = input("> ").strip().lower()

                    if user_input == "q":
                        break

                    if user_input == "1":
                        pca_ann.show_graph() if pca_ann.is_trained else \
                            alert("Error: Network is not trained yet.")

                    elif user_input == "2":
                        i = input_int("Subject ID", None, 1, 15)
                        if pca_ann.is_trained:
                            pca_ann.subject_anns[i-1].show_graph()
                        else:
                            alert("Error: Network is not trained yet.")

                    elif user_input == "3":
                        pca_ann.show_eigenfaces()

                    elif user_input == "4":
                        pca_ann.show_train_faces()

                    elif user_input == "5":
                        pca_ann.show_reconstructed_faces()

                    elif user_input == "6":
                        pca_ann.show_test_faces()

                    elif user_input == "7":
                        pca_ann.show_constructed_faces()

                    else:
                        alert("Error: Invalid input.")

        # train network
        elif user_input == "5":
            if not pca_ann:
                alert("Error: No network exists to train.")
            else:
                n_epochs = input_int("Training epochs", 50, 1)
                batch_size = input_int("Batch size", 5, 1)
                pca_ann.train(n_epochs, batch_size)
                alert("Training completed.")

        # test network
        elif user_input == "6":
            if not pca_ann:
                alert("Error: No network exists to test.")
            elif not pca_ann.is_trained:
                alert("Error: Network is not trained yet.")
            else:
                pca_ann.test()
                alert("Testing completed.")
                pca_ann.test_subject_ann(0)

        # classification reports
        elif user_input == "7":
            if not pca_ann:
                alert("Error: No network exists to "
                      "produce classification reports from."
                )
            elif not pca_ann.is_trained:
                alert("Error: Network is not trained yet.")
            else:
                for i in range(1, 16):
                    clear_screen()
                    pca_ann.test_subject_ann(i - 1)
                    alert("Above: Classification report for "
                          "subject_ann {}.".format(i)
                    )

        else:
            alert("Error: Invalid input.")

if __name__ == "__main__":
    file_name = "pca_ann.sav"
    df = pd.read_csv("face_data.csv")
    pca_ann_cli(file_name, df)
