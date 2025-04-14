# from flask import Flask, render_template, request
# from ultralytics import YOLO
# import numpy as np
# from PIL import Image
# from io import BytesIO
# import base64
# import cv2
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt

# app = Flask(__name__)
# # keras 3.8.0
# # tensorflow 2.15.0
# yolo_model = YOLO('models/yolo.pt')
# tf.keras.utils.disable_interactive_logging()
# unet_model = load_model('models/UNet-2.h5', compile=False)
# maskrcnn_model = load_model('models/MaskRCNN-2.h5', compile=False)
# classification_model = load_model('models/CNN_Pancreas.h5')

# def preprocess_image(img_bytes, target_size=(256, 256)):
#     img = Image.open(BytesIO(img_bytes)).convert('L')
#     img = img.resize(target_size)
#     img_array = np.array(img) / 255.0
#     return img_array

# def preprocess_image_for_classification(img_bytes, target_size=(150, 150)):
#     img = tf.keras.preprocessing.image.load_img(BytesIO(img_bytes), target_size=target_size)
#     img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# def classify_image(img_bytes, model):
#     img_array = preprocess_image_for_classification(img_bytes)
#     pred = model.predict(img_array)
#     if pred.shape[1] == 1:
#         pred_label = 'Abnormal' if pred[0][0] > 0.5 else 'Normal'
#     else:
#         pred_label = np.argmax(pred)
#     return pred_label

# def predict_mask(img_bytes, model, threshold=0.1):
#     preprocessed_img = preprocess_image(img_bytes)
#     pred_mask = model.predict(preprocessed_img[np.newaxis, ..., np.newaxis])[0, :, :, 0]
#     pred_mask = (pred_mask > threshold).astype(np.uint8) * 255
#     return preprocessed_img, pred_mask

# def predict_yolo(img_bytes, model):
#     nparr = np.frombuffer(img_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     results = model(img_rgb)
#     yolo_pred_image = results[0].plot()
#     return yolo_pred_image

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         img = request.files['image']
#         img_bytes = img.read()

#         ct_image, unet_pred_mask = predict_mask(img_bytes, unet_model)
#         _, maskrcnn_pred_mask = predict_mask(img_bytes, maskrcnn_model)

#         yolo_pred_image = predict_yolo(img_bytes, yolo_model)
#         pred_label = classify_image(img_bytes, classification_model)

#         plt.figure(figsize=(15.5, 6))
#         plt.subplot(1, 4, 1)
#         plt.imshow(ct_image, cmap='gray')
#         plt.title("Original CT Image")
#         plt.axis('off')

#         plt.subplot(1, 4, 2)
#         plt.imshow(cv2.cvtColor(yolo_pred_image, cv2.COLOR_BGR2RGB))
#         plt.title("YOLO Detection")
#         plt.axis('off')

#         plt.subplot(1, 4, 3)
#         plt.imshow(unet_pred_mask, cmap='gray')
#         plt.title("UNet Segmentation")
#         plt.axis('off')

#         plt.subplot(1, 4, 4)
#         plt.imshow(maskrcnn_pred_mask, cmap='gray')
#         plt.title("Mask R-CNN Segmentation")
#         plt.axis('off')

#         buf = BytesIO()
#         plt.savefig(buf, format='png', bbox_inches='tight')
#         buf.seek(0)
#         encoded_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
#         buf.close()
#         plt.close()

#         return f'<p>Pancreas is in <span style="color:red;">{pred_label}</span> condition.</p><p>Segmentation Results:</p><img src="data:image/png;base64,{encoded_plot}" class="img-fluid"/>'

# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, render_template, request
from ultralytics import YOLO
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load models
yolo_model = YOLO('models/yolo.pt')
tf.keras.utils.disable_interactive_logging()
unet_model = load_model('models/UNet-2.h5', compile=False)
maskrcnn_model = load_model('models/MaskRCNN-2.h5', compile=False)
classification_model = load_model('models/CNN_Pancreas.h5')

def preprocess_image(img_bytes, target_size=(256, 256)):
    img = Image.open(BytesIO(img_bytes)).convert('L')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return img_array

def preprocess_image_for_classification(img_bytes, target_size=(150, 150)):
    img = tf.keras.preprocessing.image.load_img(BytesIO(img_bytes), target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify_image(img_bytes, model):
    img_array = preprocess_image_for_classification(img_bytes)
    pred = model.predict(img_array)
    if pred.shape[1] == 1:
        pred_label = 'Abnormal' if pred[0][0] > 0.5 else 'Normal'
    else:
        pred_label = np.argmax(pred)
    return pred_label

def predict_mask(img_bytes, model, threshold=0.1):
    preprocessed_img = preprocess_image(img_bytes)
    pred_mask = model.predict(preprocessed_img[np.newaxis, ..., np.newaxis])[0, :, :, 0]
    pred_mask = (pred_mask > threshold).astype(np.uint8) * 255
    return preprocessed_img, pred_mask

def predict_yolo(img_bytes, model):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    results = model(img_rgb)
    yolo_pred_image = results[0].plot()
    return yolo_pred_image

def encode_image(image, cmap=None):
    """Encodes an image to a base64 string."""
    buf = BytesIO()
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return encoded_image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img = request.files['image']
        img_bytes = img.read()

        ct_image, unet_pred_mask = predict_mask(img_bytes, unet_model)
        _, maskrcnn_pred_mask = predict_mask(img_bytes, maskrcnn_model)
        yolo_pred_image = predict_yolo(img_bytes, yolo_model)
        pred_label = classify_image(img_bytes, classification_model)

        # Encode images separately
        encoded_ct_image = encode_image(ct_image, cmap='gray')
        encoded_yolo_image = encode_image(cv2.cvtColor(yolo_pred_image, cv2.COLOR_BGR2RGB))
        encoded_unet_mask = encode_image(unet_pred_mask, cmap='gray')
        encoded_maskrcnn_mask = encode_image(maskrcnn_pred_mask, cmap='gray')

        # html_output = f"""
        # <p>Pancreas is in <span style="color:red;">{pred_label}</span> condition.</p>
        # <p>Segmentation Results:</p>
        # <div style="display: flex; flex-wrap: wrap;">
        #     <div>
        #         <p>Original CT Image:</p>
        #         <img style="max-width:100%; max-height: 300px;" src="data:image/png;base64,{encoded_ct_image}" class="img-fluid"/>
        #     </div>
        #     <div>
        #         <p>YOLO Detection:</p>
        #         <img style="max-width:100%; max-height: 300px;" src="data:image/png;base64,{encoded_yolo_image}" class="img-fluid"/>
        #     </div>
        #     <div>
        #         <p>UNet Segmentation:</p>
        #         <img style="max-width:100%; max-height: 300px;" src="data:image/png;base64,{encoded_unet_mask}" class="img-fluid"/>
        #     </div>
        #     <div>
        #         <p>Mask R-CNN Segmentation:</p>
        #         <img style="max-width:100%; max-height: 300px;" src="data:image/png;base64,{encoded_maskrcnn_mask}" class="img-fluid"/>
        #     </div>
        # </div>
        # """

        # return html_output
        html_output = f"""
            <h4>Pancreas is in <span style="color:red;">{pred_label}</span> condition.</h4>
        """

        if pred_label != "Normal":
            html_output += f"""
                <p>Segmentation Results:</p>
                <div style="display: flex; flex-wrap: wrap;">
                    <div style="margin:auto;">
                        <p>Original CT Image:</p>
                        <img style="max-width:100%; max-height: 300px;" src="data:image/png;base64,{encoded_ct_image}" class="img-fluid"/>
                    </div>
                    <div style="margin:auto;">
                        <p>YOLO Detection:</p>
                        <img style="max-width:100%; max-height: 300px;" src="data:image/png;base64,{encoded_yolo_image}" class="img-fluid"/>
                    </div>
                    <div style="margin:auto;">
                        <p>UNet Segmentation:</p>
                        <img style="max-width:100%; max-height: 300px;" src="data:image/png;base64,{encoded_unet_mask}" class="img-fluid"/>
                    </div>
                    <div style="margin:auto;">
                        <p>Mask R-CNN Segmentation:</p>
                        <img style="max-width:100%; max-height: 300px;" src="data:image/png;base64,{encoded_maskrcnn_mask}" class="img-fluid"/>
                    </div>
                </div>
            """

        return html_output


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=10000)