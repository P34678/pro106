import cv2


# Create our body classifier
fullbody_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Initiate video capture for video file
vid = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while(True):
   
    # Capture the video frame by frame
    ret, frame = vid.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces, eyes and smile
    fullbody = fullbody_cascade.detectMultiScale(gray, 1.1, 5)



    # Draw the rectangle around the face, eyes and mouth 
    for (x, y, w, h) in fullbody:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

   
        
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit Window by Spacebar Key
    if cv2.waitKey(25) == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()