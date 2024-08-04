# > IMPORT DE BIBLIOTECAS

import cv2
import pygame
import numpy as np
import pandas as pd
import mediapipe as mp
from threading import Thread
from matplotlib import pyplot as plt

# -----------------------------------------------------------------------------
# > DEFININDO VARIÁVEIS

WEBCAM = 0
ALARME_ON = False
JANELA_MEDIA = 10
TEMPO_ALARME = 2200 # 2s do áudio + 200ms de gap até o looping
MEDIA_ABERTURA_PADRAO = 0.3
CONTADOR_QUADROS_SONOLENCIA = 1
QNT_FRAMES_CONSECUTIVOS_ALARME_ON = 40

# Defina variáveis para plot do gráfico
media_abertura_olhos = []
distancias_olho_direito = []
distancias_olho_esquerdo = []

# Crie um DataFrame para armazenar os dados
dados = pd.DataFrame(columns=["Média_Abertura_Olhos"])

# Definição dos pontos oculares
OLHO_DIREITO = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
OLHO_ESQUERDO = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Criando um objeto para a detecção de pontos de referência faciais
mp_face_mesh = mp.solutions.face_mesh

# Abrindo minha webcam
cap = cv2.VideoCapture(WEBCAM)


# -----------------------------------------------------------------------------
# > DEFININDO FUNÇÕES PARA O PROGRAMA

def calcular_altura_olhos(pontos):
    # Calcule as distâncias euclidianas entre os pontos de referência
    A = np.linalg.norm(np.array(pontos[15]) - np.array(pontos[1]))
    B = np.linalg.norm(np.array(pontos[14]) - np.array(pontos[2]))
    C = np.linalg.norm(np.array(pontos[13]) - np.array(pontos[3]))
    D = np.linalg.norm(np.array(pontos[12]) - np.array(pontos[4]))
    E = np.linalg.norm(np.array(pontos[11]) - np.array(pontos[5]))
    F = np.linalg.norm(np.array(pontos[10]) - np.array(pontos[6]))
    G = np.linalg.norm(np.array(pontos[9]) - np.array(pontos[7]))
    H = np.linalg.norm(np.array(pontos[0]) - np.array(pontos[8]))

    # Calcule o EAR
    altura_olhos = (A + B + C + D + E + F + G) / (2 * H)

    return altura_olhos


# >> Função para gerar o alerta
pygame.init()
ALARME = pygame.mixer.Sound('alarme.wav')


def alerta_sonoro():
    while media_abertura_olhos < MEDIA_ABERTURA_PADRAO:
        ALARME.play()  # Toque o som
        pygame.time.wait(TEMPO_ALARME)


# -----------------------------------------------------------------------------
# > LÓGICA PARA O DETECTOR DE FADIGA

# Parâmetros do Mediapipe - valores pdrões da biblioteca
with mp_face_mesh.FaceMesh(

        # Número máximo de faces a serem detectadas
        max_num_faces=1,
        # Melhora da precisão na detecção dos pontos
        refine_landmarks=True,
        # Nível mín de confiança necessário para considerar a detecção de uma face válida
        min_detection_confidence=0.5,
        # Nível mín de confiança necessário para rastrear os pontos de referência faciais ao longo do tempo
        min_tracking_confidence=0.5) as face_mesh:

    # Looping principal
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Obter o quadro atual e coletar informações da imagem
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]

        # Coletar os resultados da imagem fornecida no vídeo
        resultados = face_mesh.process(rgb_frame)

        # Condição: Se o Mediapipe foi capaz de encontrar pontos de referência no quadro...
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

            # Calcular as distâncias dos olhos direito e esquerdo
            distancia_olho_direito = calcular_altura_olhos(olho_direito)
            distancia_olho_esquerdo = calcular_altura_olhos(olho_esquerdo)

            # Armazenar as distâncias dos olhos nas listas correspondentes
            distancias_olho_direito.append(distancia_olho_direito)
            distancias_olho_esquerdo.append(distancia_olho_esquerdo)

            # Manter um histórico limitado para o cálculo da média
            if len(distancias_olho_direito) > JANELA_MEDIA:
                distancias_olho_direito.pop(0)
            if len(distancias_olho_esquerdo) > JANELA_MEDIA:
                distancias_olho_esquerdo.pop(0)

            # Calcular a média das distâncias dos olhos individualmente
            media_olho_direito = np.mean(distancias_olho_direito)
            media_olho_esquerdo = np.mean(distancias_olho_esquerdo)

            # Calcular a média da abertura dos olhos
            media_abertura_olhos = (media_olho_direito + media_olho_esquerdo) / 2

            # Atualizando o DataFrame
            dados = pd.concat([dados, pd.DataFrame({"Média_Abertura_Olhos": [media_abertura_olhos]})],
                              ignore_index=True)

            # Desenhar gráfico em tempo real na tela
            plt.clf()
            plt.plot(dados["Média_Abertura_Olhos"])
            plt.title("Média de Abertura dos Olhos")
            plt.xlabel("Quadros")
            plt.ylabel("Abertura Média")
            plt.grid()
            plt.pause(0.001)

            # Exibir na tela a média de abertura e altura dos olhos (individualmente)
            # >> Média de abertura
            cv2.putText(img=frame,
                        text=f'Media abertura: {media_abertura_olhos:.2f}',
                        fontFace=0,
                        org=(208, 20),
                        fontScale=0.7, thickness=2, color=(255, 105, 65))

            # >> Olho direito
            cv2.putText(img=frame,
                        text=f'Olho Esquerdo: {media_olho_esquerdo:.2f}',
                        fontFace=0,
                        org=(10, 450),
                        fontScale=0.5, thickness=2, color=(0, 215, 255))

            # >> Olho esquerdo
            cv2.putText(img=frame,
                        text=f'Olho Direito: {media_olho_direito:.2f}',
                        fontFace=0,
                        org=(10, 470),
                        fontScale=0.5, thickness=2, color=(0, 215, 255))

            # Condição: se os olhos estiverem meio fechados, contar.
            if media_abertura_olhos < MEDIA_ABERTURA_PADRAO:
                CONTADOR_QUADROS_SONOLENCIA += 1

                # Condição: Definindo como o alarme será tocado.
                if CONTADOR_QUADROS_SONOLENCIA > QNT_FRAMES_CONSECUTIVOS_ALARME_ON:

                    # Aviso sonoro
                    if not ALARME_ON:
                        ALARME_ON = True
                        t = Thread(target=alerta_sonoro)
                        t.deamon = True
                        t.start()

                    # Informativo visual
                    cv2.putText(img=frame,
                                text='[ALERTA] FADIGA!',
                                fontFace=0,
                                org=(238, 45),
                                fontScale=0.6, thickness=3, color=(0, 0, 255))

            else:
                CONTADOR_QUADROS_SONOLENCIA = 0
                ALARME_ON = False

        # Abrindo janela de visualziação
        cv2.imshow('Detector de fadiga - Igor Moreira', frame)

        # Precionando a tecla "ESC" o programa finaliza
        key = cv2.waitKey(1)
        if key == 27:   # 27 é o código da tecla "Esc" na tabela ASCII
            plt.close()
            break

# Salvando os dados em um arquivo CSV para futuras análises
dados.to_csv('dados_abertura_olhos.csv', index=False)

cap.release()
cv2.destroyAllWindows()

# -----------------------------------------------------------------------------
# > PLOTANDO OS RESULTADOS COLETADOS ANTERIORMENTE

# Plotando os dados aramazenados no arquivo gerado
df = pd.read_csv("dados_abertura_olhos.csv")
plt.figure()
plt.plot(dados["Média_Abertura_Olhos"])
plt.title("Gráfico Final - Média de Abertura dos Olhos")
plt.xlabel("Quadros")
plt.ylabel("Abertura Média")
plt.grid()
plt.show()
