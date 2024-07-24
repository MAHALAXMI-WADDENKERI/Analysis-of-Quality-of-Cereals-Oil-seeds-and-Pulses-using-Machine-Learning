import sqlite3
from tkinter import *
from tkinter.filedialog import askopenfilename # Open dialog box
import numpy as np
import cv2, time
from PIL import ImageTk, Image
import os


root = Tk()
root.geometry('1366x768')
root.title("Seed")

canv = Canvas(root, width=1366, height=768, bg='white')
canv.grid(row=2, column=3)
img = Image.open('back1.png')
photo = ImageTk.PhotoImage(img)
canv.create_image(2,2, anchor=NW, image=photo)


def readimg():
    filename = askopenfilename(filetypes=[("images", "*.*")])
    img = cv2.imread(filename)
    conn = sqlite3.connect('Form.db')
    cursor = conn.cursor()
    cursor.execute('delete from imgsave')
    cursor.execute('INSERT INTO imgsave(img ) VALUES(?)', (filename,))

    conn.commit()
    cv2.imshow("Seed", img)  # I used cv2 to show image
    cv2.waitKey(0)
def preprocessing():
    os.system('python preprocessing.py')
def qlt():
    Q1 = StringVar()

    conn = sqlite3.connect('Form.db')
    with conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM imgsave")
            rows = cursor.fetchall()
            for row in rows:
                filename1 = row[0]

    img = Image.open(filename1)
    width, height = img.size
    print("Dimensions:", img.size, "Total pixels:", width * height)
    pc = width * height
    img = cv2.imread(filename1, 0)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
        # find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([blur], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1, 256):
            p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
            q1, q2 = Q[i], Q[255] - Q[i]  # cum sum of classes
            if q1 < 1.e-6 or q2 < 1.e-6:
                continue
            b1, b2 = np.hsplit(bins, [i])  # weights
            # finding means and variances
            m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
            v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2
            # calculates the minimization function
            fn = v1 * q1 + v2 * q2
            if fn < fn_min:
                fn_min = fn
                thresh = i
        # find otsu's threshold value with OpenCV function
    ret, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("{} {}".format(thresh, ret))
    print(thresh)
    print(ret)
    th2 = int(thresh) * int(ret)
    th2 = int(th2) / 100
    print("Threshold value", int(thresh) * int(ret))
    th1 = int(thresh) * int(ret)
    conn = sqlite3.connect('Form.db')
    with conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM imgsave1")
            rows = cursor.fetchall()
            for row in rows:
                xxx = row[0]
    if (int(thresh) > 0) and (int(thresh) < 100):
            print("Quality is Low")
            Q1.set("Quality of " + xxx + " is Low")
    else:
            if (int(thresh) > 100) and (int(thresh) < 150):
                print("Medium")
                Q1.set("Quality of " + xxx + " is Medium")
            else:
                if (int(thresh) > 150) and (int(thresh) < 200):
                    print("High")
                    Q1.set("Quality of " + xxx + " is High")



    t1 = Entry(root, textvar=Q1,width=40, font=("BOLD", 15))
    t1.place(x=600, y=600)



def clf():
    from tkinter import messagebox

    import pandas as pd
    import os
    import sqlite3
    from mlxtend.evaluate import accuracy_score
    from skimage.transform import resize
    from skimage.io import imread
    import numpy as np
    import matplotlib.pyplot as plt

    Categories = ['Cereals', 'OilSeeds', 'Pulses']
    flat_data_arr = []  # input array
    target_arr = []  # output array
    datadir = 'dataset'
    # path which contains all the categories of images
    for i in Categories:

        print(f'loading... category : {i}')
        path = os.path.join(datadir, i)
        for img in os.listdir(path):
            img_array = imread(os.path.join(path, img))
            img_resized = resize(img_array, (150, 150, 3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(Categories.index(i))
        print(f'loaded category:{i} successfully')
    flat_data = np.array(flat_data_arr)
    target = np.array(target_arr)
    df = pd.DataFrame(flat_data)  # dataframe
    df['Target'] = target
    x = df.iloc[:, :-1]  # input data
    y = df.iloc[:, -1]  # output data
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV

    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.1, 1], 'kernel': ['rbf', 'poly']}
    svc = svm.SVC(probability=True)
    model = GridSearchCV(svc, param_grid)
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=70, stratify=y)
    print('Splitted Successfully')
    model.fit(x_train, y_train)
    print('The Model is trained well with the given images')
    # model.best_params_ contains the best parameters obtained from GridSearchCV
    y_pred = model.predict(x_test)
    print("The predicted Data is :")
    print(y_pred)
    print("The actual data is:")
    print(np.array(y_test))
    print(f"Accuracy is {accuracy_score(y_pred, y_test) * 100}% ")
    conn = sqlite3.connect('Form.db')
    with conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM imgsave")
        rows = cursor.fetchall()
        for row in rows:
            filename = row[0]
    url = filename
    img = imread(url)
    plt.imshow(img)
    plt.show()
    img_resize = resize(img, (150, 150, 3))
    l = [img_resize.flatten()]
    probability = model.predict_proba(l)
    for ind, val in enumerate(Categories):
        print("")  # print(f'{val} = {probability[0][ind] * 100}%')
    print("The predicted image is : " + Categories[model.predict(l)[0]])
    conn = sqlite3.connect('Form.db')
    cursor = conn.cursor()
    cursor.execute('delete from imgsave1')
    cursor.execute('INSERT INTO imgsave1(clff ) VALUES(?)', (Categories[model.predict(l)[0]],))

    conn.commit()
    from IPython.conftest import get_ipython

    from sklearn.metrics import classification_report
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    import mahotas
    import cv2
    import os
    import h5py

    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier

    # In[3]:

    # make a fix file size
    fixed_size = tuple((500, 500))

    # train path
    train_path = "data/"

    # no of trees for Random Forests
    num_tree = 100

    # bins for histograms
    bins = 8

    # train_test_split size
    test_size = 0.10

    # seed for reproducing same result
    seed = 9

    # In[4]:

    # features description -1:  Hu Moments

    def fd_hu_moments(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature = cv2.HuMoments(cv2.moments(image)).flatten()
        return feature

    # In[5]:

    # feature-descriptor -2 Haralick Texture

    def fd_haralick(image):
        # conver the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Ccompute the haralick texture fetature ve tor
        haralic = mahotas.features.haralick(gray).mean(axis=0)
        return haralic

    def fd_histogram(image, mask=None):
        # conver the image to HSV colors-space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # COPUTE THE COLOR HISTPGRAM
        hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
        # normalize the histogram
        cv2.normalize(hist, hist)
        # return the histog....
        return hist.flatten()

    # get the training data labels
    train_labels = os.listdir(train_path)

    # sort the training labesl
    train_labels.sort()
    print(train_labels)

    # empty list to hold feature vectors and labels
    global_features = []
    labels = []

    i, j = 0, 0
    k = 0

    # num of images per class
    images_per_class = 80

    # ittirate the folder to get the image label name

    get_ipython().run_line_magic('time', '')
    # lop over the training data sub folder

    for training_name in train_labels:
        # join the training data path and each species training folder
        dir = os.path.join(train_path, training_name)

        # get the current training label
        current_label = training_name

        k = 1
        # loop over the images in each sub-folder
        print(dir)
        for file in os.listdir(dir):

            file = dir + "/" + os.fsdecode(file)

            # read the image and resize it to a fixed-size
            image = cv2.imread(file)

            if image is not None:
                image = cv2.resize(image, fixed_size)
                fv_hu_moments = fd_hu_moments(image)
                fv_haralick = fd_haralick(image)
                fv_histogram = fd_histogram(image)
            # else:
            # print("image not loaded")

            # image = cv2.imread(file)
            # image = cv2.resize(image,fixed_size)

            # Concatenate global features
            global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

            # update the list of labels and feature vectors
            labels.append(current_label)
            global_features.append(global_feature)

            i += 1
            k += 1
        print("[STATUS] processed folder: {}".format(current_label))
        j += 1

    print("[STATUS] completed Global Feature Extraction...")

    # In[30]:

    get_ipython().run_line_magic('time', '')
    # get the overall feature vector size
    print("[STATUS] feature vector size {}".format(np.array(global_features).shape))

    # get the overall training label size
    print("[STATUS] training Labels {}".format(np.array(labels).shape))

    # encode the target labels
    targetNames = np.unique(labels)
    le = LabelEncoder()
    target = le.fit_transform(labels)
    print("[STATUS] training labels encoded...{}")
    # normalize the feature vector in the range (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(global_features)
    print("[STATUS] feature vector normalized...")

    print("[STATUS] target labels: {}".format(target))
    print("[STATUS] target labels shape: {}".format(target.shape))

    # save the feature vector using HDF5
    h5f_data = h5py.File('output/data.h5', 'w')
    h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

    h5f_label = h5py.File('output/labels.h5', 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(target))

    h5f_data.close()
    h5f_label.close()

    print("[STATUS] end of training..")

    # In[10]:

    # import the feature vector and trained labels

    h5f_data = h5py.File('output/data.h5', 'r')
    h5f_label = h5py.File('output/labels.h5', 'r')

    global_features_string = h5f_data['dataset_1']
    global_labels_string = h5f_label['dataset_1']

    global_features = np.array(global_features_string)
    global_labels = np.array(global_labels_string)

    # In[11]:

    # split the training and testing data
    (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                              np.array(global_labels),
                                                                                              test_size=test_size,
                                                                                              random_state=seed)

    # <h3>RandomForest</h3>

    # create the model - Random Forests
    clf = RandomForestClassifier(n_estimators=100)

    # fit the training data to the model
    clf.fit(trainDataGlobal, trainLabelsGlobal)

    # print(clf.fit(trainDataGlobal, trainLabelsGlobal))

    clf_pred = clf.predict(trainDataGlobal)
    # clf_pred = clf.predict(global_feature.reshape(1,-1))[0]
    print("KNN Report")
    print(classification_report(trainLabelsGlobal, clf_pred))

    test_path = "data/test"

    # loop through the test images
    # for file in glob.glob(test_path + "/*.jpg"):
    for file in os.listdir(test_path):
        file = test_path + "/" + file
        # print(file)

        # read the image
        image = cv2.imread(file)

        # resize the image
        image = cv2.resize(image, fixed_size)

        # Global Feature extraction
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick = fd_haralick(image)
        fv_histogram = fd_histogram(image)

        # Concatenate global features

        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        # predict label of test image
        prediction = clf.predict(global_feature.reshape(1, -1))[0]

        # show predicted label on image
        cv2.putText(image, train_labels[prediction], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        # display the output image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(trainDataGlobal, trainLabelsGlobal)

    clf_pred = model.predict(trainDataGlobal)
    print("KNN Report")
    print(classification_report(trainLabelsGlobal, clf_pred))
    x = [53, 56, 80]
    y = ["RF", "KNN", "SVM"]
    plt.bar(x, y)
    plt.title("Accuracy Graph")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.show()

def featext():
    import sqlite3

    import cv2
    # Reading color image as grayscale
    from skimage.filters import prewitt_h, prewitt_v

    conn = sqlite3.connect('Form.db')
    with conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM imgsave")
        rows = cursor.fetchall()
        for row in rows:
            filename = row[0]

    # reading the image
    image2 = cv2.imread(filename, 0)

    from skimage import filters, feature

    # prewitt kernel
    pre_hor = prewitt_h(image2)
    pre_ver = prewitt_v(image2)

    # Sobel Kernel
    ed_sobel = filters.sobel(image2)

    # canny algorithm
    can = feature.canny(image2)

    cv2.imshow("img", pre_ver)
    cv2.waitKey(0)
def seg():
        os.system('python seg.py')


Button(root, text='Input Image', width=20, bg='green', fg='white',  height=5,font=("bold", 10),command=readimg).place(x=700, y=300)
Button(root, text='Preprocessing', width=20, bg='green', fg='white', height=5, font=("bold", 10),command=preprocessing).place(x=872, y=300)
Button(root, text='Segmentation', width=20, bg='green', fg='white',height=5,  font=("bold", 10),command=seg).place(x=700, y=390)
Button(root, text='Feature Extraction', width=20, bg='green', fg='white',height=5, command=featext,  font=("bold", 10)).place(x=872, y=390)
Button(root, text='Classification', width=20, bg='green', fg='white',height=5, command=clf ,font=("bold", 10)).place(x=700, y=470)
Button(root, text='Quality Analysis', width=20, bg='green', fg='white', height=5, command=qlt,font=("bold", 10)).place(x=872, y=470)
root.mainloop()
