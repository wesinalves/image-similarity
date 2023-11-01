# Image similarity estimation using python - Part II
## siamese network with triplet loss
# Author: Wesin Ribeiro

# Setup
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import datasets

# Hyperparameters
batch_size = 16
epochs = 10
margin = 1

# load mnist dataset
(x_train_val, y_train_val), (x_test, y_test) = datasets.mnist.load_data()
x_train_val = x_train_val.astype("float32")
x_test = x_test.astype("float32")

# break train and val datasets
x_train, x_val = x_train_val[:30000], x_train_val[30000:]
y_train, y_val = y_train_val[:30000], y_train_val[30000:]

# create triplets
def make_triplets(x,y):
    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    triplets = []

    for idx1 in range(len(x)):
        a = x[idx1]
        label1 = y[idx1]
        p = random.choice(digit_indices[label1])

        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)
        
        n = random.choice(digit_indices[label2])

        triplets += [[a,p,n]]
    
    return np.array(triplets)

triplets_train = make_triplets(x_train)
triplets_val = make_triplets(x_val)
triplets_test = make_triplets(x_test)

fake_labels_train = np.zeros((len(y_train), 30)).astype("float32")
fake_labels_val = np.zeros((len(y_val), 30)).astype("float32")
fake_labels_test = np.zeros((len(y_test), 30)).astype("float32")

print(triplets_train.shape)

# visualize data
def visualize(anchor, positive, negative):
    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    fig = plt.figure(figsize=(5,5))
    axs = fig.subplots(3,3)
    for i in range(3):
        show(axs[i,0], anchor[i])
        show(axs[i,1], positive[i])
        show(axs[i,2], negative[i])

# split triplets in singular vectors
x_train_a = triplets_train[:,0]
x_train_p = triplets_train[:,1]
x_train_n = triplets_train[:,2]
out_train = fake_labels_train[:]

x_val_a = triplets_val[:,0]
x_val_p = triplets_val[:,1]
x_val_n = triplets_val[:,2]
out_val = fake_labels_val[:]

x_test_a = triplets_test[:,0]
x_test_p = triplets_test[:,1]
x_test_n = triplets_test[:,2]
out_test = fake_labels_test[:]

visualize(x_train_a, x_train_p, x_train_n)

# define triplet loss function

def loss(margin=1):
    def triplet_loss(y_true, y_pred):
        a = y_pred[:, 0:10]
        p = y_pred[:, 10:20]
        n = y_pred[:, 20:30]

        pos_dist = tf.math.reduce_sum(tf.math.square(a-p), axis=1)
        neg_dist = tf.math.reduce_sum(tf.math.square(a-n), axis=1)

        loss = pos_dist - neg_dist
        loss = tf.maximum(loss + margin, 0)

        return loss
    return triplet_loss

# embedding network
input = layers.Input((28,28,1))
x = layers.BatchNormalization()(input)
x = layers.Conv2D(4, (5,5), activation="tanh")(x)
x = layers.AvaregePooling2D(pool_size=(2,2))(x)
x = layers.Conv2D(16, (5,5), activation="tanh")(x)
x = layers.AvaregePooling2D(pool_size=(2,2))(x)
x = layers.Flatten()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(10, activation="tanh")(x)
embedding_network = keras.Model(input,x)
embedding_network.summary()

# siamese network
input_a = layers.Input((28,28,1))
input_p = layers.Input((28,28,1))
input_n = layers.Input((28,28,1))

tower_a = embedding_network(input_a)
tower_b = embedding_network(input_p)
tower_n = embedding_network(input_n)

merge_layer = layers.Concatenate([tower_a, tower_b, tower_n])
siamese = keras.Model([input_a, input_p, input_n], merge_layer)
siamese.summary()

# compile and train
siamese.compile(loss=loss(margin), optimizer="adam", metrics="mse")

history = siamese.fit(
    [x_train_a, x_train_p, x_test_n],
    out_train,
    validation_data=([x_val_a,x_val_p,x_val_n], out_val),
    batch_size = batch_size,
    epochs = epochs
)

# plot learning curve
def plot_metric(history, metric, title, has_valid=True):
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    
    plt.title(title)
    plt.xlabel("epochs")
    plt.ylabel(metric)
    plt.show()

plot_metric(history.history, "loss", "triple loss")

# check predictions

visualize(x_test_a, x_test_p, x_test_n)

def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), keepdims=True)
    dist = tf.math.sqrt(tf.maximum(sum_square, tf.keras.bakend.epsilon()))
    return dist

anchor_embedding = embedding_network(x_test_a)
positive_embedding = embedding_network(x_test_p)
negative_embedding = embedding_network(x_test_n)

pos_similarity = euclidean_distance([anchor_embedding, positive_embedding])
print("Positive distance: ", pos_similarity.numpy())

neg_similarity = euclidean_distance([anchor_embedding, negative_embedding])
print("Positive distance: ", neg_similarity.numpy())





































































