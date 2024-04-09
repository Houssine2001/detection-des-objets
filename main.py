import cv2
import mysql.connector

# Fonction pour se connecter à la base de données MySQL
def connect_to_database():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="pidevgymweb"
        )
        return connection
    except mysql.connector.Error as error:
        print("Error while connecting to MySQL", error)
        return None

# Fonction pour vérifier si le nom de l'objet est présent dans la table "produit"
def check_object_in_database(object_name, connection):
    cursor = connection.cursor()
    query = "SELECT * FROM produit WHERE nom = %s"
    cursor.execute(query, (object_name,))
    result = cursor.fetchone()
    cursor.close()
    if result:
        return True
    else:
        return False

# Fonction principale pour détecter les objets depuis la caméra et vérifier dans la base de données
def Camera():
    # Connexion à la base de données
    connection = connect_to_database()
    if connection is None:
        print("Failed to connect to database.")
        return

    cam = cv2.VideoCapture(0)
    
    # Ajuster la taille de la capture vidéo
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    classNames = []
    classFile = 'coco.names'

    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightpath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightpath, configPath)
    net.setInputSize(320, 230)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    while True:
        success, img = cam.read()
        classIds, confs, bbox = net.detect(img, confThreshold=0.5)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                object_name = classNames[classId - 1]
                print(object_name)

                # Vérifier si l'objet est présent dans la base de données
                if check_object_in_database(object_name, connection):
                    print("Produit trouvé dans notre store")
                    cv2.putText(img, f"{object_name} (Produit disponible)", (box[0] + 10, box[1] + 40),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)
                else:
                    cv2.putText(img, object_name, (box[0] + 10, box[1] + 20),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)

                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)

        cv2.imshow('Output', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    connection.close()

# Appeler la fonction principale
Camera()
