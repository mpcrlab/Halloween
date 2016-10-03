from keras.models import Model
from keras.optimizers import Nadam, SGD
from data.data_utilities import load_data
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense, Dropout


# create the base pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False)

# add pooling and dense layers
net = base_model.output
net = GlobalAveragePooling2D()(net)
net = Dropout(p=0.5)(net)
net = Dense(16, activation='softmax')(net)

# build the model
model = Model(input=base_model.input, output=net)

# freeze all layers in base model
for layer in base_model.layers:
    layer.trainable = False

# compile the model
model.compile(optimizer=Nadam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# load in the data
x_train, y_train, x_valid, y_valid = load_data()
x_train = preprocess_input(x_train)
x_valid = preprocess_input(x_valid)

# fit to the data
model.fit(x_train, y_train, batch_size=32, nb_epoch=30, validation_data=(x_valid, y_valid))

# unfreeze some of the later layers
for layer in model.layers[:162]:
    layer.trainable = False
for layer in model.layers[162:]:
    layer.trainable = True

# compile the model again
model.compile(
    optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy']
)
model.summary()

# fit to the data again
model.fit(x_train, y_train, batch_size=32, nb_epoch=50, validation_data=(x_valid, y_valid))
