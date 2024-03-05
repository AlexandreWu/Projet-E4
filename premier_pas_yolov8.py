from ultralytics import YOLO

# Charger le modèle YOLO
model = YOLO('yolov8n.pt')



# Paramètres de suivi avec les classes restreintes
results = model.track(source='suitcase.mp4', show=True, tracker="bytetrack.yaml", classes = [0,28,24,26], save=True, name='test', save_txt = True)

