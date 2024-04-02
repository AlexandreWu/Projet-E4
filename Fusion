import os
import datetime
import pandas as pd
import cv2
from ultralytics import YOLO
from pymongo import MongoClient

# Initialisation de l'environnement et de la base de données
def initialize_database():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    client = MongoClient('mongodb://localhost:27017/')
    client.drop_database('tracking_database')
    print("Base de données 'tracking_database' supprimée.")
    db = client['tracking_database']
    return db['humans'], db['suitcases'], db['associations']

# Fonction pour calculer la distance euclidienne entre deux boîtes
def calculate_distance(bbox1, bbox2):
    center1 = (bbox1[0] + bbox1[2] / 2, bbox1[1] + bbox1[3] / 2)
    center2 = (bbox2[0] + bbox2[2] / 2, bbox2[1] + bbox2[3] / 2)
    return ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5

# Suivi et association des objets dans la vidéo
def track_and_associate(video_path, model, humans_collection, suitcases_collection):
    class_ids = {'person': 0, 'suitcase': 28, 'backpack': 24}
    results = model.track(source=video_path, show=True, tracker="bytetrack.yaml", classes=list(class_ids.values()), save=True, name='tracking_results', stream=True)
    
    for i, frame_result in enumerate(results):
        process_frame(frame_result, i, humans_collection, suitcases_collection)

# Traiter chaque frame pour la détection et l'association
def process_frame(frame_result, frame_number, humans_collection, suitcases_collection):
    class_ids = {'person': 0, 'suitcase': 28, 'backpack': 24}
    timestamp = datetime.datetime.utcnow()
    for box in frame_result.boxes:
        track_id, class_id, bbox, conf = box.id.item(), box.cls.item(), box.xywh.squeeze().tolist(), box.conf.item()
        coord = box.xyxy[0].tolist()
        detection_doc = {'track_id': track_id, 'class_id': class_id, 'bbox': bbox, 'conf': conf, 'timestamp': timestamp, 'coord': coord, 'frame': frame_number}
        
        # Mise à jour des collections en fonction de la classe de l'objet
        target_collection = humans_collection if class_id == class_ids['person'] else suitcases_collection
        target_collection.update_one({'track_id': track_id}, {'$set': detection_doc}, upsert=True)

# Identifier les bagages abandonnés basé sur la logique définie
def identify_abandoned_luggage(associations_collection, threshold_distance=300, threshold_movement=6, max_frames_without_movement=1000):
    print("Identification des bagages abandonnés en cours...")
    abandoned_bag_track_ids = set()
    association_docs = list(associations_collection.find({}))

    last_positions = {}
    frames_without_movement = {}

    for doc in sorted(association_docs, key=lambda x: x['frame']):
        track_id = doc['object_id']
        bbox = doc['bbox_object']
        if track_id not in last_positions:
            last_positions[track_id] = bbox
            frames_without_movement[track_id] = 0
            continue

        distance_moved = calculate_distance(last_positions[track_id], bbox)
        if distance_moved < threshold_movement:
            frames_without_movement[track_id] += 1
        else:
            frames_without_movement[track_id] = 0

        if frames_without_movement[track_id] >= max_frames_without_movement:
            abandoned_bag_track_ids.add(track_id)

        last_positions[track_id] = bbox

    print(f"Bagages abandonnés détectés : {abandoned_bag_track_ids}")
    return abandoned_bag_track_ids

# Générer des alertes sur la vidéo pour les bagages abandonnés
def generate_alerts(video_path, abandoned_bag_track_ids, associations_collection):
    cap, output_video = prepare_video_output(video_path)
    process_video_frames(cap, output_video, abandoned_bag_track_ids, associations_collection)
    finalize_video_output(cap, output_video)

def prepare_video_output(video_path):
    # Initialiser la capture vidéo et le writer de vidéo de sortie
    cap = cv2.VideoCapture(video_path)
    fps, frame_width, frame_height = get_video_properties(cap)
    output_video = cv2.VideoWriter('Video_with_alerts.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))
    return cap, output_video

def get_video_properties(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, frame_width, frame_height

def process_video_frames(cap, output_video, abandoned_bag_track_ids, associations_collection):
    color = (0, 0, 255)  # Rouge en BGR pour l'alerte
    thickness = 2  # Épaisseur du rectangle
    font = cv2.FONT_HERSHEY_SIMPLEX  # Police pour le texte

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        for track_id in abandoned_bag_track_ids:
            doc = associations_collection.find_one({'object_id': track_id, 'frame': current_frame})
            if doc:
                coord = doc['coord_object']
                x1, y1, x2, y2 = map(int, coord)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, 'Abandonné', (x1, y1 - 10), font, 0.5, color, 2)

        output_video.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    finalize_video_output(cap, output_video)

def finalize_video_output(cap, output_video):
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()
    print("Vidéo avec alertes sauvegardée.")
    
    
# Main function to encapsulate the program logic
def main():
    humans_collection, suitcases_collection, associations_collection = initialize_database()
    model = YOLO('yolov8n.pt')
    video_path = 'test.mp4'

    track_and_associate(video_path, model, humans_collection, suitcases_collection)
    abandoned_bag_track_ids = identify_abandoned_luggage(associations_collection)
    generate_alerts(video_path, abandoned_bag_track_ids, associations_collection)

if __name__ == "__main__":
    main()
