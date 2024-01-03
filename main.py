# Face-Recognition használata

import face_recognition
import matplotlib.pyplot as plt

known_image = face_recognition.load_image_file("reference.image.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

unknown_image01 = face_recognition.load_image_file("image.02.jpg")
unknown_image02 = face_recognition.load_image_file("image.01.jpg")

face_locations_1 = face_recognition.face_locations(unknown_image01)
face_encodings_1 = face_recognition.face_encodings(unknown_image01, face_locations_1)

face_locations_2 = face_recognition.face_locations(unknown_image02)
face_encodings_2 = face_recognition.face_encodings(unknown_image02, face_locations_2)

for face_encoding in face_encodings_1:
    matches_1 = face_recognition.compare_faces([known_face_encoding], face_encoding)


for face_encoding in face_encodings_2:
    matches_2 = face_recognition.compare_faces([known_face_encoding], face_encoding)


plt.imshow(unknown_image01)
if True in matches_1:
    plt.title("Tamas ott van!")
else:
    plt.title("Tamas nincs ott!")
plt.show()

plt.imshow(unknown_image02)
if True in matches_2:
    plt.title("Tamas ott van!")
else:
    plt.title("Tamas nincs ott!")
plt.show()