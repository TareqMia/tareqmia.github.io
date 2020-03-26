import cv2 as cv

original_image = cv.imread("image.jpg")
grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY) #convert image to grayscale
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_alt.xml")
dected_faces = face_cascade.detectMultiScale(grayscale_image, 1.3)


#draws rectangle around the found faces
for (column, row, width, height) in dected_faces:
    cv.rectangle(
                original_image,
                 (column, row),
                 (column + width, row + height),
                 (255,0,0),
                 3)

#resize image
small = cv.resize(original_image, (0,0), fx=0.2, fy=0.2)
cv.imshow("Image", small)
cv.waitKey(0)
cv.destroyAllWindows()







