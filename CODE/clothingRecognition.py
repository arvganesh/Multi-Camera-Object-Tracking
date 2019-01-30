from config import *

def detect_clothes(img): # Detects clothing in an image
    clth_model = load_model(PATH_TO_MODEL) # Load Model
    img = transform(img)
    pred = clth_model.predict(img)
    return pred

def transform(img): # image path
    im = keras_image.load_img(img, target_size=(200, 200))
    im = keras_image.img_to_array(im)   
    im = np.expand_dims(im, axis=0)        
    # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    im /= 255.     
    return im

print (detect_clothes("jeans.jpeg"))