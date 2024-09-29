import cv2
import face_recognition

# Load the images of Rachel, Phoebe, Chandler, Ross, Monica, and Joey to learn their faces
rachel_image = face_recognition.load_image_file("Rachel.jpg")
phoebe_image = face_recognition.load_image_file("Phoebe.jpg")
chandler_image = face_recognition.load_image_file("Chandler.jpg")
ross_image = face_recognition.load_image_file("Ross.jpg")
monica_image = face_recognition.load_image_file("Monica.jpg")
joey_image = face_recognition.load_image_file("Joey.jpg")

# Encode the faces for recognition
rachel_encoding = face_recognition.face_encodings(rachel_image)[0]
phoebe_encoding = face_recognition.face_encodings(phoebe_image)[0]
chandler_encoding = face_recognition.face_encodings(chandler_image)[0]
ross_encoding = face_recognition.face_encodings(ross_image)[0]
monica_encoding = face_recognition.face_encodings(monica_image)[0]
joey_encoding = face_recognition.face_encodings(joey_image)[0]

# Create a list of known face encodings and their corresponding names
known_face_encodings = [
    rachel_encoding,
    phoebe_encoding,
    chandler_encoding,
    ross_encoding,
    monica_encoding,
    joey_encoding
]

known_face_names = [
    "Rachel",
    "Phoebe",
    "Chandler",
    "Ross",
    "Monica",
    "Joey"
]

# Load the image of the group (Friends.jpg)
img = cv2.imread("Friends.jpg")

# Convert the image to RGB (as face_recognition works with RGB format)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Find all face locations and face encodings in the current image
face_locations = face_recognition.face_locations(rgb_img)
face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

# Loop through each face in this image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known faces
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"  # Default name if no match is found

    # Use the first match found to label the face
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    # Draw a rectangle around the face
    cv2.rectangle(img, (left, top), (right, bottom), (135, 135, 112), 3)

    # Draw the name of the person above the rectangle
    cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

# Display the image with recognized faces and their names
cv2.imshow("Friends - Face Recognition with Names", img)

# Wait for the user to close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()
