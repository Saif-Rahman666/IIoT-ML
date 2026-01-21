import tensorflow as tf

def convert_lstm_standard(model_path, output_path):
    # 1. Load the model
    model = tf.keras.models.load_model(model_path)
    
    # 2. Setup the converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 3. STATIC SHAPE FIX: Replace None with our actual values
    # Input shape: [Batch_Size, Sequence_Length, Number_of_Features]
    # We use 1 for batch size, 30 for sequence, 8 for features
    batch_size = 1
    sequence_length = 30
    num_features = 8
    
    # Force the input shape to be static
    model.input.set_shape((batch_size, sequence_length, num_features))
    
    # 4. Conversion Settings
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter._experimental_lower_tensor_list_ops = True
    
    try:
        tflite_model = converter.convert()
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        print(f"🔥 SUCCESS: {output_path} created without Flex Ops!")
    except Exception as e:
        print(f"❌ Conversion failed: {e}")

# Run the conversion
convert_lstm_standard('models/lstm_rul_model.h5', 'models/lstm_rul.tflite')