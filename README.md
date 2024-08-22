# <p align="center"> Face Detection & Recognition System </p>
<p align="center"> Count the number of Known and Unknown people in a specific area from video stream. </p>


<p align="center">
  <img src="https://github.com/ShathaAltassan/FDR-System/assets/138797663/09541351-1a96-47f8-a43c-9a90068dd1a8" alt="FDR Logo">
</p>

## Description



This code is for a face recognition system that interacts with a MongoDB database to store reference face embeddings along with their names. It imports necessary libraries such as OpenCV, NumPy, PyTorch, and others. Establishes a connection to a MongoDB database. Loads pre-trained face detection and face recognition models. Fetching Reference Data: Retrieves reference face embeddings and associated names from the MongoDB database. Recognition Function: Defines a function to recognize faces based on cosine similarity between reference embeddings and embeddings of detected faces. Uses a distance threshold to determine if a detected face matches a reference face. Main Loop for Face Recognition: Takes a video file path as input. Reads frames from the video and detects faces in each frame using the MTCNN model. Processes each detected face by resizing, preprocessing, and obtaining its embedding using the InceptionResnetV1 model. Matches the embeddings with reference embeddings to recognize known faces or classify them as unknown. Updates counters for known and unknown faces. Draws bounding boxes around detected faces and displays recognition results on the video frames. Continues processing until the user quits by pressing 'e'. Cleanup and Resource Release: Clears recognized embeddings for privacy. Releases video resources and closes windows. This code essentially performs real-time face recognition using pre-trained models and stores/retrieves reference embeddings from a MongoDB database.



## Key Features

- **Image Upload**: Users can upload images containing faces for recognition.
- **Video Processing**: The application processes uploaded videos frame by frame to detect and recognize faces.
- **Live Streaming**: Users can stream live video from a camera for real-time face recognition.
- **Database Integration**: MongoDB is used for storing face embeddings and associated data, enabling seamless data management and retrieval.

## Models Used

- **MTCNN (Multi-Task Cascaded Convolutional Networks)**: This model is utilized for face detection, providing bounding box coordinates for detected faces.
- **InceptionResnetV1**: Used for face recognition, extracting numerical embeddings of faces for comparison and recognition tasks.

## Functionality Overview

1. **Image Upload**:
   - Users can upload images through the application's user interface.
   - Uploaded images are processed using the MTCNN model to detect faces and extract embeddings.
   - Extracted embeddings are stored in MongoDB along with associated metadata (e.g., name or ID).
     

2. **Video Processing**:
   - Uploaded videos are processed frame by frame.
   - Each frame is analyzed using the MTCNN model to detect faces and the InceptionResnetV1 model to extract embeddings.
   - Recognition logic compares extracted embeddings with stored embeddings to identify known and unknown faces.
     

3. **Live Streaming**:
   - The application supports live streaming from a camera source.
   - Live video feed undergoes real-time face detection and recognition, providing instantaneous recognition results.

## Implementation

- **Database**:
MongoDB is used as the database for storing face embeddings and associated data. It enables efficient data management and retrieval, supporting the core functionality of the FDR System application.

Here is how the MongoDB connected with Visual Studio Code

![image](https://github.com/AfrahSaud36/FDR-System/assets/138797663/e592d008-8113-4f10-888e-f53e19c7a3e6)

![mongoserver](https://github.com/AfrahSaud36/FDR-System/assets/138797663/8dd06de9-a9e1-40f3-a755-5fb1059fc3cf)

- **Technical architecture of the application**:

The navigation diagrams of the actors are showed through figure. The navigation diagram provides a description of the system's structure, how to design interfaces, navigate between them, and how to connect objects

![Picture2](https://github.com/AfrahSaud36/FDR-System/assets/138797663/28732c66-55ea-4703-aeb3-2e8224bb885b)

- **How our system works**:
  
  ![image](https://github.com/AfrahSaud36/FDR-System/assets/138797663/5b452c0e-e4ca-418d-bb03-4c282d1a2b2a)


  ## Final Interfaces of the application

  
Home Page

![image](https://github.com/AfrahSaud36/FDR-System/assets/138797663/cb3ad2e4-1232-4186-8465-7c2261e10032)

About and Contact Page

![image](https://github.com/AfrahSaud36/FDR-System/assets/138797663/7ecd0c9f-797f-4633-ae63-a65f6a3e747f)

FDR System Page

![image](https://github.com/AfrahSaud36/FDR-System/assets/138797663/a1ef83f3-c527-4702-ad02-f42973153518)

Main Sections

![image](https://github.com/AfrahSaud36/FDR-System/assets/138797663/1952570e-c97c-4831-b6c7-847aadf9d1e0)



## Testing (Result)

 **Database Building Process Results**
 
 ![image](https://github.com/AfrahSaud36/FDR-System/assets/138797663/4b5eafa9-a237-4ccf-a593-c69965f9f550)

 •	The system successfully provides the user with options to either create a new database or use an existing one, enhancing flexibility and usability.
 
 •  Users are able to input the desired name for the new database and specify the number of people to include, allowing for customization according to their requirements

 
![image](https://github.com/AfrahSaud36/FDR-System/assets/138797663/73e30545-9468-41fd-85c8-3804f5caf336)

* The interface for uploading face images is user-friendly, enabling users to easily upload images for each individual.
* Users can associate names with the uploaded images, facilitating the organization and identification of individuals within the database.

![image](https://github.com/AfrahSaud36/FDR-System/assets/138797663/fafdbd2a-4d18-4b29-b8e2-9107e18383ab)

•	MongoDB integration seamlessly stores both the names and vector embeddings of the uploaded face images.

• The database structure efficiently manages the data, ensuring retrieval and manipulation processes are optimized for performance


![image](https://github.com/AfrahSaud36/FDR-System/assets/138797663/e3ec8d3e-6848-4542-a517-719f54cfe58b)

![image](https://github.com/AfrahSaud36/FDR-System/assets/138797663/5308a158-d6e8-4660-8694-237c2044a588)

•	Throughout the database building process, the system demonstrates reliability by accurately saving uploaded data without errors or inconsistencies.

•	Stability is maintained even with varying amounts of data input, indicating robustness in handling different user requirements.


**Uploaded Video Process Results**

![image](https://github.com/AfrahSaud36/FDR-System/assets/138797663/2fa9f806-c1b0-4324-a6d2-5123ea29a2bb)


![image](https://github.com/AfrahSaud36/FDR-System/assets/138797663/86967e3c-17fd-4853-905c-162feb45fb23)

* During testing, this process was completed successfully, indicating that our system performs as intended. Whether it's tallying recognized individuals or identifying unknown faces, the system consistently delivers the desired outcome. This success underscores the system's capability to meet the specified requirements and demonstrates its effectiveness as a whole. By seamlessly integrating into the user's workflow and providing reliable results, our face counting system proves its value in various applications, promising practical utility in real-world.


## Code

**Face Embedding Extraction**

* The faces are detected and cropped using the MTCNN (Multi-task Cascaded Convolutional Networks).
* The face embeddings are generated using the InceptionResnetV1 model pre-trained on the VGGFace2 dataset.

  ```python

  # Detect faces and extract embeddings
  faces, _ = mtcnn(img, return_prob=True)

  if faces is not None:
    # Get embeddings
    embeddings = resnet(faces).detach().cpu().numpy()

**Recognition Process**

* The recognize_face function is used to compare the extracted face embeddings with reference embeddings stored in the MongoDB database.
* The comparison is done using the cosine similarity measure. If the similarity exceeds a threshold, the face is recognized.

  ```python
       def recognize_face(reference_data, query_embedding, threshold=0.7):
       for reference_embedding, name in reference_data:
        # Resize reference embedding to match query embedding dimensionality
        reference_embedding_resized = np.resize(reference_embedding, query_embedding.shape)
        similarity = 1 - cosine(reference_embedding_resized, query_embedding)
        if similarity >= threshold:
            return True, name, similarity
      return False, None, None

**Integration in Video Processing**

* Faces are detected in each frame of the video.
* For each detected face, embeddings are generated and compared with the stored reference embeddings using the recognize_face function.
* The results are used to draw bounding boxes and labels around recognized faces in the video frames.
  
    ```python 
        # Main loop for processing video frames
        while cap.isOpened():
            # Capture frame from video
            ret, frame = cap.read()
            frame_count += 1
        
            # Skip frames if necessary
            if frame_count % frame_skip != 0:
                continue
        
            if not ret:
                break
        
            # Detect faces in frame
            boxes, _ = mtcnn.detect(frame)
        
            # If faces are detected, process each one
            if boxes is not None:
                for box in boxes:
                    # Extract face from frame
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    face = frame[y1:y2, x1:x2]
        
                    # Preprocess query image
                    query_image = cv2.resize(face, (160, 160))
                    query_image = query_image / 255.0
                    query_embedding = resnet(torch.tensor(query_image, dtype=torch.float).unsqueeze(0).permute(0, 3, 1, 2).to(device)).squeeze().detach().cpu().numpy()
        
                    # Recognize face with distance threshold
                    recognized, name, similarity = recognize_face(reference_data, query_embedding, threshold=distance_threshold)
        
                    # Draw bounding box around face
                    color = (0, 255, 0) if recognized else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, name if recognized else "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)



## Conclusion

We created a face detection and recognition system to solve the challenge of rapidly and reliably counting known and unknown persons in a predetermined region. We used the MTCNN model to identify faces and the Inception-ResNet-v1 model to recognize faces. 
During the project's execution, we experienced several issues, including the challenge of selecting an acceptable model to design the system and then tying it to the web application. We aim to have done the finest and smoothest work possible. While conducting testing, the system will accomplish the predicted results of face detection and identification in educational situations

## Demo 

![Demo-min](https://github.com/AfrahSaud36/FDR-System/assets/138797663/3fc043e2-dd91-4f22-b5eb-bc97b10bf62b)
