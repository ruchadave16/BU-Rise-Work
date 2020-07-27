# Machine Learning - Supervised learning, a classification task
# Exercise based on Chapter 3 of ...

#  packages to support python 3
from __future__ import division, print_function, unicode_literals

# Other packages that we need to import for use in our script
import numpy as np
import os

# to make this script's output stable across runs, we need to initialize the random number generator to the same starting seed each time
np.random.seed(42)

# To plot nice figures
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures that our code generates
def save_fig(fig_id, tight_layout=True):
    path = os.path.join("ML_" + fig_id + ".png") #PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

####################
# STEP 1: OBTAIN THE DATA
####################

# Now, we need to download a dataset to work with
# and load it into our workspace:
import pickle

def save_obj(obj, name="mnist" ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name="mnist"  ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='bytes')

df = load_obj()

# Now, If we want to display part of the dataset on the screen, we can simply
# enter the name of the variable we called it:

print(df)

# TO DO: We can see that our variable is a dictionary. It contains multiple 
# fields that are named and have different types of data in them. If we want
# a clearer look at just the field names in the dictionary, we can use the 
# "keys()" function by entering our variable name followed immediately by 
# ".keys()"  (but without the quotes around it):
# ?.keys()

print(df.keys())

# TO DO: let's organize our dataset into two variables, one a matrix that contains
#        all the descriptions of the handwritten numbers (in the "data" key of the
#        dictionary) and one a vector that contains the labels for each of those
#        numbers, the label being the actual number that was written (in the "target"
#        key of the dictionary).
#        Let's name the matrix of handwriting data "X" and the vector of labels "y":
X = df[b'data']
y = df[b'target']

# TO DO: Now, we can check the size of our dataset by using the shape function (this is
#        one of the few times where we need to call it without adding parenthesis at the
#        the end). We can call this function on our dataset by adding ".shape" to the
#        variable name:
print(X.shape)

# Now we see that our matrix has 70,000 rows, meaning it describes 70,000
# handwritten numbers. The matrix has 728 columns, so each of the 70,000
# handwritten numbers in our database is described by 728 data points. These points
# define a 28 pixel x 28 pixel square. There is one data point for each pixel in the
# square, which gives the shade of gray in that square (anything from white to black
# and all the shades of gray in-between). If we were to graph a figure using the data
# for one of those handwritten numbers, we would get a 28x28 sized square that depicts
# a single hand-written number.

####################
# STEP 2: EXPLORE THE DATA
####################

# TO DO: Import the plotting package matplotlib. Uncomment the following
#        two lines to import matplotlib and to have a short way to refer
#        to its subpackage pyplot:
import matplotlib
import matplotlib.pyplot as plt

# TO DO: Let's look at one of the numbers described in our matrix. First, define a new
#        variable called 'some_digit' that is equal to one of the rows of data in matrix X:
some_digit = X[0] #Call the first row

# TO DO: Now, we need to reshape that row into a square, the 28x28 square that can
#        represent the picture of that handwritten number. We can do this using the
#        "reshape" function, which takes two arguments, the row and column dimensions
#        to reshape our variable into:
some_digit_image = some_digit.reshape(28, 28)

# TO DO: Next, we need to use pyplot's "imshow" function to show our image. The arguments
#        it takes are the 28x28 matrix we just made to describe the image, a colormap that
#        tells pyplot which colors are represented by which values found in the matrix, and
#        a method for handling values in the matrix that don't exactly match the values
#        defined in the color map (ex: if the color map only says 0.1 = almost black and
#        0.5 = medium gray, pyplot needs to know what to do for the value 0.3). Uncomment
#        the following lines to show a picture of the handwritten number:
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,interpolation="nearest")
plt.axis("off")

# TO DO: What if we want to look at other numbers in our dataset? It's more efficient
#        to write a function that repeats the steps we just did:
#        Given a row of the matrix, the function should reshape it into a 28x28 square,
#        plot it in an image using the imshow function, and then turn the axis off so
#        that the resulting figure looks like an image and not a graph. Try defining a
#        function called 'plot_digit' that accomplishes this task, where 'data' is the
#        name given the incoming argument (the row of the matrix with 728 data points).
#        Remember to indent the substatements of the function by four spaces:
def plot_digit(data):
	img = data.reshape(28, 28)
	plt.imshow(img, cmap = matplotlib.cm.binary, interpolation="nearest")
	plt.axis("off")


# TO DO: Now, let's test out our new function on the handwritten number in row
#        3600 of our X matrix. We should be able to call it as follows:
plot_digit(X[36000])

# What number do you see when you call that function? Is the number hard to read?
# What number is the computer likely to think it is? If you are unsure what number
# it is, can you think of an easy way to find out?

#This is a 5. The image is kind of blurry

# TO DO: Sometimes we may want to view a whole bunch of numbers at once. For that,
#        we can define a different function that shows many numbers together. The
#        function is already written for you here, just uncomment each line. As you
#        uncomment each line, take a look at it to see if you can figure out what
#        that line of code is doing:
def plot_digits(instances, images_per_row=10, **options):
	size = 28
	images_per_row = min(len(instances), images_per_row)
	images = [instance.reshape(size,size) for instance in instances]
	n_rows = (len(instances) - 1) // images_per_row + 1
	row_images = []
	n_empty = n_rows * images_per_row - len(instances)
	images.append(np.zeros((size, size * n_empty)))
	for row in range(n_rows):
		rimages = images[row * images_per_row : (row + 1) * images_per_row]
		row_images.append(np.concatenate(rimages, axis=1))
	image = np.concatenate(row_images, axis=0)
	plt.imshow(image, cmap = matplotlib.cm.binary, **options)
	plt.axis("off")


# TO DO: Now, we can make use of this new function to show many handwritten
#        numbers at a time. Uncomment the following lines to show many numbers:
plt.figure(figsize=(9,9))
example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
plot_digits(example_images, images_per_row=10)



####################
# STEP 2: SEPARATE TRAINING AND TEST DATA
####################

# TO DO: We need to separate both our number data matrix (X) and
#        the vector of labels that contains the true identity of
#        each number (y). We have 70,000 numbers total, so lets
#        use 60,000 numbers for training and 10,000 for test.
#        Use your knowledge of slicing arrays to select the first
#        60,000 numbers for training and the last 10,000 for testing:
X_train = X[:60000]
X_test = X[60000:]
y_train = y[:60000]
y_test = y[60000:]


# Next, let's shuffle the data in case there is an order to the
# numbers in our matrix. Machine learning algorithms may be thrown
# off if a lot of the same number are presented in a row or if the
# order of numbers presented follows a specific pattern. Therefore,
# we will randomly rearrange the numbers in our test data set to
# disrupt any patterns in the presentation of the numbers.

# TO DO: Uncomment the following code to load in a new module and
#        create a vector that will list the row indices of our
#        test matrix in random order. Then we can use that vector
#        to rearrange the test matrix:
import numpy as np

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]



####################
# STEP 3: TRAIN THE NETWORK
####################

# We'll first start with a simpler task of having the network
# only differentiate between 5s and everything else.

# We need to copy the rows from our data matrix that correspond
# to the number 5. However, the only way to know which rows
# represent 5s is to consult our label vector. We can create a
# vector of row indices that correspond to 5s from our label
# vector using the commands below. Notice that we have to do
# this both for the training vector and for the test vector.

# TO DO: Uncomment the code below to create vectors that
#        list whether each entry in the training and test sets
#        is a 5 or is not a 5:
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)


# Now, we are ready to train our network using the training
# data. The algorithm we want to use, SGDClassifier, must
# be imported from the sklearn module.

# TO DO: Uncomment the code below to import the algorithm:
from sklearn.linear_model import SGDClassifier

# TO DO: Now, we need to call the SGDClassifier. This function
#        requires two named arguments, max_iter and random_state.
#        The max_iter argument sets the maximum number of
#        iterations to perform while learning the task (it is
#        a coincidence that this number should be set to 5). The
#        random_state argument lets us specify a random seed
#        to use to initialize our network. Lets use 42. Set the
#        output of the function to 'sgd_clf'. This represents
#        our learning algorithm object.
#
sgd_clf = SGDClassifier(max_iter = 5, random_state = 42)

# TO DO: Next, we need to provide our learning algorithm with the
#        training data and its corresponding labels. Because we
#        created a new vector called y_train_5, our labels for this
#        task are no longer the actual number represented by the
#        training data, but are instead all set to 0 for numbers
#        that are not 5 and to 1 for numbers that are 5.  Call the
#        fit() method of the sgd_clf object with two arguments, the
#        training data set and the new label vector (with 0 for non-5
#        and 1 for 5).
sgd_clf.fit(X_train, y_train_5)


# TO DO: Now let's see what our trained algorithm thinks that strange
#        looking number is. Uncomment and run the following code to
#        find out:
some_digit = X_train[43679]
print(sgd_clf.predict([some_digit])) # Results True - so it thinks the code is 5

####################
# STEP 5: EVALUATE THE NETWORK PERFORMANCE
####################

# To DO: Now we want to measure the performance of the training algorithm
#        taking into account all of the training data. We can find out the
#        exact performance of the algorithm because we know the real identities
#        of the number samples and we know what the algorithm came up with
#        for each sample. We need to import the function 'cross_val_score'
#        from sklearn.model_selection and then call that function with the
#        arguments 'sgd_clf' (the learning algorithm object), our training
#        dataset, our label vector that labels whether each training row
#        is a 5 or not a 5 (called y_train_5, don't use the other vector y_train
#        for this exercise), and two named arguments: cv=3, scoring="accuracy"
from sklearn.model_selection import cross_val_score
print("cross_val_score = ", cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))


# Bonus: if you want to google for other ways of measuring the performance, go for it!
# Otherwise, we will proceed further with this exercise tomorrow:
# - Refining the model, if necessary
# - Running the model on the test data
# - Calculating overall model performance on test data

####################
# STEP 5: EVALUATE THE NETWORK PERFORMANCE
# CONTINUED FROM YESTERDAY (Paste this code into your previous
# script file - this file won't run on its own.)
####################################

# After reading about confusion matrices in your lab manual, let's return
# to this code and create a confusion matrix for our data.


# First, we need to see how our model performs on the training data:
from sklearn.model_selection import cross_val_predict
y_train_pred_5 = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# Now, we compare the model's predictions with our actual labels:
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train_5, y_train_pred_5))

# We can also generate a precision score and a recall score:
from sklearn.metrics import precision_score, recall_score
print("precision = ", precision_score(y_train_5, y_train_pred_5))
print("recall = ", recall_score(y_train_5, y_train_pred_5))

# And we can generate an f1 score that takes both precision
# and recall into effect:
from sklearn.metrics import f1_score
print("f1 = ", f1_score(y_train_5, y_train_pred_5))


y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,  method="decision_function")

y_scores.shape


# Note: there was an [issue](https://github.com/scikit-learn/scikit-learn/issues/9589)
# in Scikit-Learn 0.19.0 (fixed in 0.19.1) where the result of `cross_val_predict()`
# was incorrect in the binary classification case when using
# `method="decision_function"`, as in the code above. The resulting array
# had an extra first dimension full of 0s. Just in case you are using
# 0.19.0, we need to add this small hack to work around this issue:
if y_scores.ndim == 2:
    y_scores = y_scores[:, 1]


# Now, we can look at how precision and recall performance vary
# as a function of the algorithm's threshold:
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-700000, 700000])
save_fig("precision_recall_vs_threshold_plot")

# TODO: Questions based on the graph - you can discuss these
# with your group - how do precision and recall depend on
# threshold? 
#
# What is the relationship between precision and recall?
#
# Why do you think there is that relationship?


def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
save_fig("precision_vs_recall_plot")

# Note:
# In a real machine learning use-case, we would optimize
# our model based on the performance of the algorithm
# on our training data. 
#
# This is one reason why it is helpful to split the
# training data into multiple subsets
#
# Another reason may be to develop an algorithm separately
# for each training dataset, and then apply a hybrid
# of the algorithms developed on each subset and use
# that hybrid of algorithms as your overall algorithm
# going forward and for test data. That way, you can
# reduce the effect of any particular bias in algorithm
# or subset of training data


####################
# STEP 6: USE OUR TEST DATA
####################

# Once we are satisfied with the performance of our
# algorithm on the other subsets of training data,
# we can apply our algorithm to our test dataset.

# TODO: Let's run the fitted model on our test data.
# Run the cross_val_predict function on our test
# data subsets that we set aside earlier:
y_test_scores = cross_val_predict(sgd_clf, X_test, y_test_5, cv=3, method="decision_fu)


# TODO: Now, using some of the model performance calculations
# above, characterize the performance of the algorithm
# on the test dataset. Remember that, for some of our code,
# we already defined a plotting function so we can call that
#same function with the appropriate test variables to get our output:

test_precisions, test_recalls, test_thresholds = precision_recall_curve(y_test_5, y_test_scores)

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(test_precisions, test_recalls, test_thresholds)
plt.xlim([-700000, 700000])
save_fig("precision_recall_vs_threshold_plot_test")
plt.show()

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(test_precisions, test_recalls)
save_fig("precision_vs_recall_plot_test")

# Hint: when you calculate the confusion matrix, precision, recall, f1
# you may want to print them to the console to ensure they show up in
# the output


# TODO: Now, try running a multivariate classification instead
# of the binary classification we have done so far (5 versus non-5)
# To do this, you could make a copy of code you've
# already worked through above, starting from where you recoded your
# outputs as 5 or not 5. What were the original y and x variables called?
# You can substitute in those variables in the training, prediction, and
# performance calculation commands that follow.
# Hint: start by creating a new classifier object and training it:
sgd_clf_multi = SGDClassifier(max_iter=5, random_state=42)
sgd_clf_multi.fit = sgd_clf_multi.fit(X_train, y_train)
some_digit = X_train[43679]
print(sgd_clf_multi.predict(some_digit))

# With multivariate classification, it's possible for the classifier
# to work by comparing a possible label with any other one possible lable,
# and doing many 1:1 comparisons to figure out the label of one input.
# Or the classifier can work by having a competition amongst all possible
# labels. To explicitly set a particular method that the classifier
# should use, you can import additional methods from sklearn and
# pass your classifier to them:


# TODO: Obtain the confusion matrix, recall, precision, and f1
# scores for the multivariate classifier
#
# Hint: to obtain the precision, recall, and f1 stats for the
# multivariate classifier, you need to pass in an additional
# argument to each of those functions, `average='weighted'`


# TODO: when you create your confusion matrix for the multivariate
# classifier, it will be much larger than the binary classifier matrix!
# You may choose to plot it for quicker comprehension:
# conf_mx = ...
#
# After you create your confusion matrix and assign it to variable conf_mx
# You can uncomment the following code:
# plt.matshow(conf_mx, cmap=plt.cm.gray)
# plt.show()

# We can improve this plot. First, let's normalize the
# number of errors by the number of images in each category
# so that abundant classes don't appear overly error-prone
# (ie, plot the rate of error rather than the absolute number).
# Uncomment the following code:
# row_sums = conf_mx.sum(axis=1, keepdims=True)
# norm_conf_mx = conf_mx / row_sums
#
#
# And then we can also fill the diagonal (which contains the
# number/rate of correct instances with 0s so that we are only
# plotting the error rates. Uncomment the following code.
# 
# TODO: look up the help for the method np.fill_diagonal using
# help(), ?, or just googling. How can you use that method to
# fill the diagonal with 0s? 
#
# np.fill_diagonal ... 

# TODO: Call the plot commands to plot and show the updated
# confusion matrix:


