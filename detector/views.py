from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import io




def pneumonia(request):
    return render(request,'pneumonia.html')

@csrf_exempt
@require_POST
def predict_pneumonia(request):
    # Load your trained .h5 model
    model = tf.keras.models.load_model('ml_models/pneumonia_detection_model.h5')

    # Process the incoming image
    image_data = request.FILES['image']  # Assuming the image is sent as a file in a POST request
    img = image.load_img(io.BytesIO(image_data.read()), target_size=(224, 224))  # Read and preprocess as in your training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale pixel values to [0, 1]

    # Make predictions
    predictions = model.predict(img_array)

    # Interpret the predictions
    if predictions[0] > 0.5:
        result = "Infected (Pneumonia)"
    else:
        result = "Not Infected (Normal)"

    return JsonResponse({'result': result})