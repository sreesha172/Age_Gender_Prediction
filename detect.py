import cv2
import matplotlib.pyplot as plt
import os

# Paths to model files
model_dir = "C:/Users/sreer/OneDrive/Desktop/Data Science Projects/age gender prediction/"
faceProto = os.path.join(model_dir, "opencv_face_detector.pbtxt")
faceModel = os.path.join(model_dir, "opencv_face_detector_uint8.pb")
ageProto = os.path.join(model_dir, "age_deploy.prototxt")
ageModel = os.path.join(model_dir, "age_net.caffemodel")
genderProto = os.path.join(model_dir, "gender_deploy.prototxt")
genderModel = os.path.join(model_dir, "gender_net.caffemodel")

# Check if model files exist
model_files = [faceProto, faceModel, ageProto, ageModel, genderProto, genderModel]
for file in model_files:
    if not os.path.isfile(file):
        print(f"Error: Model file not found: {file}")
        exit()

# Load models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Mean values for model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Categories
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# List of input image paths
image_paths= [
    'C:/Users/sreer/OneDrive/Desktop/Data Science Projects/age gender prediction/kid.jpg',
    'C:/Users/sreer/OneDrive/Desktop/Data Science Projects/age gender prediction/women.jpg',
    'C:/Users/sreer/OneDrive/Desktop/Data Science Projects/age gender prediction/man.jpg',
    'C:/Users/sreer/OneDrive/Desktop/Data Science Projects/age gender prediction/old.jpg',
]

# Prompt the user for the image name
target_image_name = input("Enter the image name (e.g., kid1.jpg): ")

# Find the target image path based on the input image name
target_image_path = None
for image_path in image_paths:
    if os.path.basename(image_path) == target_image_name:
        target_image_path = image_path
        break

if target_image_path is None:
    print(f"Error: Image '{target_image_name}' not found in the list.")
    exit()

# Process the target image
image = cv2.imread(target_image_path)

# Check if the image was successfully loaded
if image is None:
    print(f"Error: Could not load image at path: {target_image_path}")
    exit()

image = cv2.resize(image, (720, 640))

# Copy image
frame = image.copy()

# Prepare input blob
frameHeight = frame.shape[0]
frameWidth = frame.shape[1]
blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)

# Detect faces
faceNet.setInput(blob)
detections = faceNet.forward()

# Detect face bounding boxes
faceBoxes = []
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.7:
        x1 = int(detections[0, 0, i, 3] * frameWidth)
        y1 = int(detections[0, 0, i, 4] * frameHeight)
        x2 = int(detections[0, 0, i, 5] * frameWidth)
        y2 = int(detections[0, 0, i, 6] * frameHeight)
        faceBoxes.append([x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)

# Check if faces are detected
if not faceBoxes:
    print("No face detected in image:", target_image_path)
    exit()

# Predict age and gender for each face
for faceBox in faceBoxes:
    face = frame[max(0, faceBox[1]-15):min(faceBox[3]+15, frame.shape[0]-1),
                 max(0, faceBox[0]-15):min(faceBox[2]+15, frame.shape[1]-1)]
    
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    
    # Predict gender
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]
    
    # Predict age
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]
    
    # Draw the predicted age and gender
    cv2.putText(frame, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

# Display the output
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title(f'Results for {os.path.basename(target_image_path)}')
plt.axis('off')
plt.show()
