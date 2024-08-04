# > IMPORT DE BIBLIOTECAS

import cv2
import numpy as np
import mediapipe as mp


# -----------------------------------------------------------------------------
# > DEFININDO VARIÁVEIS

WEBCAM = 0

# Localização dos pontos de referência dos olhos na coleção do FaceMesh.
OLHO_DIREITO = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
OLHO_ESQUERDO = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Criando um objeto para a detecção de pontos de referência faciais
mp_face_mesh = mp.solutions.face_mesh

# Abrindo minha webcam
cap = cv2.VideoCapture(WEBCAM)


# -----------------------------------------------------------------------------
# > DETECTANDO CONTORONO DOS OLHOS + CONTORNO DO ROSTO

# Parâmetros do Mediapipe - valores pdrões da biblioteca
with mp_face_mesh.FaceMesh(

        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while True:

        # Obter cada quadro da webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Obter o quadro atual e coletar informações da imagem
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]

        # Coletar os resultados da imagem fornecida no vídeo
        resultados = face_mesh.process(rgb_frame)

        # Condição: Se o Mediapipe foi capaz de encontrar pontos de referência no quadro
        if resultados.multi_face_landmarks:

            # Coletar todos os pares [x, y] de todos os pontos de referência faciais
            todos_pontos_referencia = np.array(
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in
                 resultados.multi_face_landmarks[0].landmark])

            # Calcular o retângulo delimitador do rosto
            x, y, w, h = cv2.boundingRect(todos_pontos_referencia)

            # Desenhar o retângulo verde
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 205, 50), 2)

            # Pontos de referência dos olhos direito e esquerdo
            olho_direito = todos_pontos_referencia[OLHO_DIREITO]
            olho_esquerdo = todos_pontos_referencia[OLHO_ESQUERDO]

            # Desenhar somente os pontos de referência dos olhos na imagem
            cv2.polylines(frame, [olho_esquerdo], True, (152, 251, 152), 1, cv2.LINE_AA)
            cv2.polylines(frame, [olho_direito], True, (152, 251, 152), 1, cv2.LINE_AA)


        # Abrindo janela de visualziação
        cv2.imshow('Detector de fadiga - Igor Moreira', frame)
        # Precionando a tecla "ESC" o programa finaliza
        key = cv2.waitKey(1)
        if key == 27:  # 27 é o código da tecla "Esc" na tabela ASCII
            break

cap.release()
cv2.destroyAllWindows()
