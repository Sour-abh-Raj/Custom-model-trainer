from tensorflow.keras.models import model_from_json

# Load model architecture
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# Load model weights
loaded_model.load_weights('model_weights.h5')
