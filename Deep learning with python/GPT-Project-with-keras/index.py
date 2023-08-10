import keras
from keras import layers, models

# Creating a sequential model
model = keras.Sequential()
model.add(layers.Dense(units=64, activation='relu', input_shape=(input_size)))
model.add(layers.Dense(units=10, activation='softmax'))

# Compiling model with required arguments
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Getting a summary of the model
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

# Model evaluation
test_loss, test_accuracy = model.evaluate(x_test, y_test)

# Make a prediction
predictions = model.predict(x_new_data)

# Save and load model
model.save('model.h5')
model = keras.models.load_model('model.h5')
