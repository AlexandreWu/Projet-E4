from ultralytics import YOLO

# Initialize the model
model = YOLO('yolov8n.pt')

# Run tracking
results = model.track(source='suitcase_2.mp4', show=True, tracker="bytetrack.yaml", classes=[0, 28, 24, 26], save=True, name='test')

# Initialize an empty list to store tracking data
tracking_data = []

# # Process results
# for result in results:
#     for box in result.boxes:
#         # Extract tracking information
#         track_id = box.id.item()
#         class_id = box.cls.item()
#         x, y, w, h = box.xywh
#         conf = box.conf.item()
        
#         # Append to the Python list
#         tracking_data.append((track_id, class_id, x, y, w, h, conf))
        
#         # Here you can also insert the data into your database
#         # For SQL, it would be something like:
#         # cursor.execute("INSERT INTO tracking (track_id, class_id, x, y, w, h, conf) VALUES (?, ?, ?, ?, ?, ?, ?)", (track_id, class_id, x, y, w, h, conf))
#         # For MongoDB, it would be something like:
#         # collection.insert_one({"track_id": track_id, "class_id": class_id, "bbox": [x, y, w, h], "conf": conf})

# # Don't forget to commit your changes if you're using SQL
# # connection.commit()

# # Now, `tracking_data` contains all your tracking information
# print(tracking_data)

# Process results
for result in results:
    for box in result.boxes:
        # Print box.xywh to understand its format
        # print("xywh:", box.xywh)
        
        # Extract tracking information
        track_id = box.id.item()
        class_id = box.cls.item()
        # Check if box.xywh contains expected format
        # if box.xywh.dim() == 2 and box.xywh.size(1) == 4:
        x, y, w, h = box.xywh.squeeze().tolist()  # Extract inner tensor and convert to list
        conf = box.conf.item()
        tracking_data.append((track_id, class_id, x, y, w, h, conf))
        # else:
        #     print("Unexpected format for box.xywh:", box.xywh)

print (tracking_data)
