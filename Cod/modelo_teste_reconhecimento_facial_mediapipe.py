# Importa as bibliotecas necessárias
import cv2
import mediapipe as mp

# Inicializa objetos da MediaPipe para desenhar os resultados
mp_drawing = mp.solutions.drawing_utils  # Inicializa um objeto para desenhar anotações nos resultados da MediaPipe
mp_drawing_styles = mp.solutions.drawing_styles  # Estilos de desenho predefinidos para a MediaPipe
mp_face_mesh = mp.solutions.face_mesh  # Inicializa o modelo de detecção de malha facial da MediaPipe

# Configuração para desenho das anotações faciais
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)  # Configura as especificações para desenho

# Inicializa a captura de vídeo da webcam (câmera 0)
cap = cv2.VideoCapture(0)  # Inicializa a captura de vídeo da câmera padrão (câmera 0)

# Inicializa o modelo de detecção de malha facial da MediaPipe
with mp_face_mesh.FaceMesh(
        max_num_faces=1,  # Número máximo de faces a serem detectadas
        refine_landmarks=True,  # Refina os pontos de referência
        min_detection_confidence=0.5,  # Confiança mínima de detecção
        min_tracking_confidence=0.5) as face_mesh:  # Confiança mínima de rastreamento

    # Loop para processar quadros da webcam
    while cap.isOpened():  # Enquanto a captura de vídeo da webcam estiver ativa

        success, image = cap.read()  # Lê um quadro da webcam e verifica o sucesso da leitura

        if not success:
            print(
                "Ignorando quadro vazio.")
            continue

        image.flags.writeable = False  # Marca a imagem como somente leitura para otimizar o desempenho
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converte a imagem de BGR para RGB

        results = face_mesh.process(image)  # Processa a imagem com o modelo de malha facial da MediaPipe

        image.flags.writeable = True  # Marca a imagem como gravável novamente
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Converte a imagem de RGB de volta para BGR

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Desenha os pontos de referência faciais e as conexões da malha
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                # Desenha os contornos faciais
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                # Desenha as anotações dos olhos
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        # Espelha a imagem horizontalmente para exibição no modo de selfie
        cv2.imshow('MediaPipe Face Mesh',
                   cv2.flip(image, 1))  # Exibe a imagem espelhada na janela 'MediaPipe Face Mesh'

        # Verifica se a tecla 'Esc' foi pressionada para encerrar o programa
        if cv2.waitKey(1) == 27:
            break  # Se a tecla 'Esc' for pressionada, encerra o loop

# Libera a captura de vídeo da webcam
cap.release()  # Libera os recursos de captura de vídeo da câmera
