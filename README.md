# Détection et suivi d'objets dans une vidéo

Ce script Python utilise le modèle YOLO (You Only Look Once) pour détecter et suivre les humains et les valises dans une vidéo. Il identifie également les bagages abandonnés en fonction de certaines conditions définies et génère des alertes visuelles sur la vidéo.

## Introduction

Ce script utilise un modèle YOLO pré-entraîné pour détecter les humains et les valises dans une vidéo fournie. Il suit ces objets au fil des frames de la vidéo, enregistre les données de détection dans une base de données MongoDB et identifie les bagages abandonnés.

## Comment exécuter le script

Pour exécuter le script, suivez ces étapes :

1. Assurez-vous d'avoir Python 3.x installé sur votre système.
2. Installez les dépendances requises en exécutant `pip install -r requirements.txt`.
3. Assurez-vous d'avoir une instance MongoDB en cours d'exécution localement sur le port par défaut (27017).
4. Placez votre vidéo dans le même répertoire que le script et mettez à jour le chemin de la vidéo dans la variable `video_path`.
5. Exécutez le script en utilisant la commande `python script.py`.

## Paramètres personnalisables

Vous pouvez ajuster certains paramètres dans le script pour correspondre à vos besoins :

- `threshold_distance`: La distance minimale entre un humain et une valise pour être considérée comme abandonnée (par défaut: 300).
- `threshold_movement`: Le mouvement minimum d'une valise pour ne pas être considérée comme abandonnée (par défaut: 6).
- `max_frames_without_movement`: Le nombre maximum de frames qu'une valise peut rester immobile avant d'être considérée comme abandonnée (par défaut: 400).

## Remarques

- Ce script a été testé sur des vidéos de courte durée. Pour des vidéos plus longues, il est recommandé d'ajuster les paramètres en fonction de la durée et de la fréquence des mouvements dans la vidéo.
- La sortie vidéo avec les alertes sera enregistrée sous le nom "test_git.mp4" dans le répertoire de travail.
