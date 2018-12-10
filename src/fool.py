import foolbox
from foolbox.criteria import TargetClassProbability
from PIL import Image
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions
import keras
import numpy as np
import cv2


keras.backend.set_learning_phase(0)
kmodel = keras.applications.resnet50.ResNet50(weights='imagenet')
model = foolbox.models.KerasModel(kmodel, bounds=(0, 255))

image = cv2.imread('/home/chrisquion/green_thumb/noface/data/lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg', cv2.IMREAD_COLOR)
label = '234'

attack = foolbox.attacks.LBFGSAttack(model=model, criterion=TargetClassProbability(1, p=.5))
adv = attack(image, label)
cv2.imwrite("adv.jpg", adv)

print("Predicted class of original: ", np.argmax(model.predictions(image)))
print("Predicted class of adversarial: ", np.argmax(model.predictions(adv)))
print("Probability of adversarial class: ", foolbox.utils.softmax(model.predictions(adv))[1])
adv_copy = Image.open("adv.jpg")
adv_copy = adv_copy.resize((224, 224)) # For some reason this needs to be resized

preds = kmodel.predict(np.expand_dims(adv_copy, axis=0))
print("Top 5 adversarial: ", decode_predictions(preds, top=5))
