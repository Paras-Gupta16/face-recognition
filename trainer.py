import os
import cv2
import numpy as np
from PIL import Image

# Create the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "dataset"

def get_images_with_id(path):
    images_path = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    
    for single_image_path in images_path:
        faceImg = Image.open(single_image_path).convert('L')  # Convert image to grayscale
        facenp = np.array(faceImg, np.uint8)
        Id = int(os.path.split(single_image_path)[-1].split(".")[1])
        print(Id)
        faces.append(facenp)
        ids.append(Id)
        
        cv2.imshow("Training", facenp)
        cv2.waitKey(10)
    
    return np.array(ids), faces

ids, faces = get_images_with_id(path)

# Train the recognizer on the faces and ids
recognizer.train(faces, ids) 

# Save the trained model
recognizer.save("recognizer/trainingdata.yml")
cv2.destroyAllWindows()
