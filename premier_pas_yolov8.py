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

# print (tracking_data)




from pymongo import MongoClient

# Connexion à la base de données MongoDB
client = MongoClient('mongodb://localhost:27017/')  # URL de connexion MongoDB
database = client['tracking_database']  # Sélectionnez ou créez une base de données

# Créez une collection pour stocker les données de suivi
tracking_collection = database['tracking_data']

# Parcourez les données de suivi et insérez-les dans la collection MongoDB
for track_data in tracking_data:
    # Créez un document à insérer dans la collection
    track_document = {
        'track_id': track_data[0],
        'class_id': track_data[1],
        'x': track_data[2],
        'y': track_data[3],
        'w': track_data[4],
        'h': track_data[5],
        'conf': track_data[6]
    }
    # Insérez le document dans la collection MongoDB
    tracking_collection.insert_one(track_document)

# Fermez la connexion à la base de données
client.close()

print("Les données de suivi ont été stockées dans la base de données MongoDB avec succès.")
