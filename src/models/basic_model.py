from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        self.model = Sequential([
            layers.Input(shape=input_shape),

            # reduce image size
            layers.Conv2D(16, (3,3), padding='same', activation='relu'),
            # keep maximum value of 3x3 area of image
            layers.MaxPooling2D((2,2)),
            # reduce image size further, more filters for better tracking 
            layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            # batch normalization to also reduce overfitting
            layers.BatchNormalization(),
            # keep maximum value of 3x3 area to reduce image size more
            layers.MaxPooling2D((2,2)),
            # reduce image size further, more filters for better tracking 
            layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            # keep maximum value of 3x3 area to reduce image size more
            layers.MaxPooling2D((2,2)),
            
            # flatten for fully connected layer
            layers.Flatten(),

            # fully connected layer
            layers.Dense(6, activation='relu'),
            # dropout to mitigate overfitting at a 50% rate
            layers.Dropout(0.5),
            # softmax 
            layers.Dense(categories_count, activation='softmax')
            # Notes for Writeup : 
            # Increased conv2D layers by 1 to allow for higher detail and accuracy from CNN, and increased filter argument for the same purpose.
            # Added batch normalization to reduce overfitting and increase accuracy. Added dropout layer to reduce overfitting.
            # Lowered the MaxPooling2D layers from 4x4, 3x3 all down to 2x2 to cut down on loss of information and parameter usage
        ])


    def _compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001), # I may have increased the learning rate to 0.0001, I forgot the default value
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )