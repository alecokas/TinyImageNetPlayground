from tensorflow import app

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, PReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121

from utility_functions import parse_args, setup_callbacks, create_output_directory, datetime_as_string, get_current_datetime, save_model # NOQA


NUM_CLASSES = 50
IMAGE_SIZE = 64 * 2


def build_single_layer():
    """ Build a simple network with a single convolutional layer followed by a single fully connected layer.
    """
    print("Building Single Layer Network")
    model = Sequential()
    model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(Flatten())
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model


def build_transfer_learning_based_classifier():
    """ Build transfer learning based model using DenseNet121 with frozen weights which were pretrained on imagenet.
    """
    print("Building a Transfer Learning Based Network")

    pretrained_model = Sequential()
    pretrained_model = DenseNet121(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    x = pretrained_model.output
    x = Flatten()(x)

    x = Dense(units=2048, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(0.4)(x)

    predictions = Dense(NUM_CLASSES, activation="softmax")(x)

    # Freeze layers
    for layer in pretrained_model.layers:
        layer.trainable = False

    # Create the combined model
    model_final = Model(input=pretrained_model.input, output=predictions)
    return model_final


def build_cnn_classifier():
    """ Build a CNN image classifier from scratch.
    """
    print("Building a CNN network from scratch")
    model = Sequential()

    model.add(Conv2D(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                     filters=32,
                     kernel_size=(3, 3),
                     kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    # 2nd conv-pool
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(3, 3),
                           padding="same"))
    model.add(Dropout(0.2))

    # 3rd conv-pool
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(3, 3),
                           padding="same"))
    model.add(Dropout(0.2))

    # flatten the output so that it can be input to the fully connected layers
    model.add(Flatten())

    model.add(Dense(units=512,
                    kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.35))

    model.add(Dense(units=512,
                    kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.35))

    model.add(Dense(units=NUM_CLASSES,
                    activation="softmax"))

    return model


def train_models(models, model_callbacks, num_epochs, batch_size, dataset_dir, output_directory_name):
    """ Train all models sequentially and saves the weights, accuracy log, and  model architecture in
        the directory output_directory_name. model_callbacks is passed to the fit_generator function.
    """
    for model_name in models:

            model = models[model_name]

            num_images_per_class = 500
            num_train_images = NUM_CLASSES * num_images_per_class
            num_val_images = 10000

            print('Training model: ' + str(model_name))
            save_model(model, model_name, output_directory_name)

            # Instructing Keras to print model summary
            model.summary()

            # Compiling a model selecting a categorical cross-entropy loss, 'adam' optimiser and accuracy metric
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                shear_range=0.5,
                zoom_range=0.5,
                brightness_range=(0.0, 0.1),
                rotation_range=20,
                horizontal_flip=True
            )

            vaidation_datagen = ImageDataGenerator(
                rescale=1. / 255
            )

            test_datagen = ImageDataGenerator(
                rescale=1. / 255
            )

            train_generator = train_datagen.flow_from_directory(
                directory=dataset_dir + 'train',
                target_size=(IMAGE_SIZE, IMAGE_SIZE),
                color_mode='rgb',
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=True,
                seed=42
            )

            validation_generator = vaidation_datagen.flow_from_directory(
                directory=dataset_dir + 'val/images',
                target_size=(IMAGE_SIZE, IMAGE_SIZE),
                color_mode='rgb',
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=True,
                seed=42
            )

            test_generator = test_datagen.flow_from_directory(
                directory=dataset_dir + 'test',
                target_size=(IMAGE_SIZE, IMAGE_SIZE),
                color_mode='rgb',
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=True,
                seed=42
            )

            history = model.fit_generator(
                train_generator,
                steps_per_epoch=num_train_images // batch_size,
                epochs=num_epochs,
                validation_data=validation_generator,
                validation_steps=num_val_images // batch_size,
                callbacks=model_callbacks
            )

            loss, accuracy = model.evaluate_generator(
                test_generator,
                steps=None
            )

            print('==================================================')
            print('Test Accuracy: {}'.format(accuracy))
            print('==================================================')


def define_models():
    """ Define a few models which will be trained sequentially.
    """
    models = {}
    models['Transfer Learning'] = build_transfer_learning_based_classifier()
    models['CNN'] = build_cnn_classifier()
    models['Single Layer Model'] = build_single_layer()

    return models


def main(args):

    parsed_args = parse_args(args)

    output_directory_name = create_output_directory(
        output_directory=parsed_args.output_directory
    )
    callbacks = setup_callbacks(output_directory_name)
    models = define_models()

    train_models(
        models=models,
        model_callbacks=callbacks,
        num_epochs=parsed_args.num_epochs,
        batch_size=parsed_args.batch_size,
        dataset_dir=parsed_args.data_location,
        output_directory_name=parsed_args.output_directory
    )
    print("Complete")


if __name__ == "__main__":
        app.run()
