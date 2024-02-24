from PIL import Image
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/1")

# defining various terms
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50

# Importing dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "training/PlantVillage",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)
class_names = dataset.class_names

# Percentage of data for training, testing, etc
train_size = 0.8
train_ds = dataset.take(54)
test_ds = dataset.skip(54)
val_size = 0.1
val_ds = test_ds.take(6)
test_ds = test_ds.skip(6)


def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):

    ds_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# Predicting random image from dataset and
# displaying target and predicted result
for images_batch, labels_batch in test_ds.take(1):

    first_image = images_batch[0].numpy().astype('uint8')
    print(f"Output type: {type(first_image)}")
    first_label = labels_batch[0].numpy()

    print("first image to predict")
    img = Image.fromarray(first_image, 'RGB')
    # img.save('my.png')
    img.show()
    # plt.imshow(first_image)
    print("actual label:", class_names[first_label])

    batch_prediction = model.predict(images_batch)
    print("predicted label:", class_names[np.argmax(batch_prediction[0])])
