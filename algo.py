import pandas as pd
from pymongo import MongoClient
import time
import cv2

# Connexion à la base de données MongoDB
client = MongoClient('mongodb://localhost:27017/')  # URL de connexion MongoDB
database = client['tracking_database']  # Sélectionnez ou créez une base de données

# Récupérer les données de suivi depuis la collection MongoDB
tracking_collection = database['associations']
cursor = tracking_collection.find()

# Créer une liste pour stocker les données de suivi
tracking_data = []
for document in cursor:
    tracking_data.append(document)

# Fermer la connexion à la base de données
client.close()

# Créer un DataFrame à partir des données de suivi
df = pd.DataFrame(tracking_data)

# Maintenant, vous pouvez utiliser df pour traiter et analyser vos données plus facilement
# print(df.head())  # Afficher les premières lignes du DataFrame

abandoned_bag_track_ids = []


# Nombre maximal de frames sans mouvement pour déclencher l'alerte
max_frames_without_movement = 1000

# Dictionnaire pour stocker les dernières positions des valises par track_id
last_bag_positions = {}

# Dictionnaire pour compter le nombre de frames sans mouvement pour chaque valise
frames_without_movement = {}
first_abandoned_frames = {}

# Boucle sur les données de suivi
for index, row in df.iterrows():
    distance = row['distance']
    track_id = row['suitcase_id']
    bbox= row['bbox_valise']
    # x = row['x']
    # y = row['y']
    frame_number = row['frame']
    
    # Supposons que la classe 28 représente les valises
    if distance > 300:
        x= bbox[0]
        y=bbox[1]
        bag_center = (x, y)
        
        # Vérifier si le track_id de la valise a déjà été vu
        if track_id in last_bag_positions:
            # Vérifier si la position de la valise est la même que la dernière position enregistrée
            last_x, last_y = last_bag_positions[track_id]
            distance_moved = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5
            
            # Si la valise ne bouge pas ou bouge très peu, incrémenter le compteur de frames sans mouvement
            if distance_moved < 6:
                frames_without_movement[track_id] = frames_without_movement.get(track_id, 0) + 1
            else:
                frames_without_movement[track_id] = 0  # Réinitialiser le compteur si la valise bouge
                
            # Si la valise n'a pas bougé pendant un certain nombre de frames, déclencher une alerte
            if frames_without_movement[track_id] >= max_frames_without_movement:
                if track_id not in first_abandoned_frames:
                    first_abandoned_frames[track_id] = frame_number
                print(f"Alerte : Valise {track_id} abandonnée pendant {max_frames_without_movement} frames à la frame {frame_number}.")
                abandoned_bag_track_ids.append(track_id)

        else:
            # Initialiser les positions et le compteur pour un nouveau track_id de valise
            last_bag_positions[track_id] = (x, y)
            frames_without_movement[track_id] = 0

        # Mettre à jour la dernière position de la valise
        last_bag_positions[track_id] = (x, y)

# Vérifier si certaines valises n'ont pas bougé jusqu'à la fin de la vidéo
for track_id, frame_count in frames_without_movement.items():
    if frame_count >= max_frames_without_movement:
        print(f"Alerte : Valise {track_id} abandonnée jusqu'à la fin de la vidéo.")
        abandoned_bag_track_ids.append(track_id)

    



# Charger la vidéo
video_path = 'Video1.mp4'  # Mettez le chemin de votre vidéo ici
cap = cv2.VideoCapture(video_path)

# Récupérer les détails de la vidéo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)


# Définir la couleur et l'épaisseur du cadre rouge
color = (0, 0, 255)  # Rouge
thickness = 2


# Supposons que vous ayez déjà les track_id des valises abandonnées dans la liste abandoned_bag_track_ids


# Définissez le codec et créez un objet VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = 'Video1_with_alerts.mp4'  # Nom du fichier de sortie
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Boucle sur les frames de la vidéo
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Récupérer le numéro de frame actuel
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    # Dessiner un cadre rouge autour des valises abandonnées
    for index, row in df[(df['frame'] == current_frame) & (df['suitcase_id'].isin(abandoned_bag_track_ids))].iterrows():
        cords = row['coord_valise']
        x1, y1, x2, y2 = cords
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        if current_frame > first_abandoned_frames.get(row['suitcase_id'], -1):
            # Dessiner le cadre rouge autour de la valise
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, 'Alerte', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Écrire la frame modifiée dans la vidéo de sortie
    output_video.write(frame)
    
    # Afficher la frame
    cv2.imshow('Frame', frame)
    
    # Attendre le temps correspondant au fps avant d'afficher la prochaine frame
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
output_video.release()
cv2.destroyAllWindows()

print(f"La vidéo avec les alertes a été enregistrée sous {output_video_path}.")
