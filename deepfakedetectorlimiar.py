# -*- coding: utf-8 -*-
"""
Visualização ao vivo do Fluxo Óptico (Farnebäck) para vídeo com REAL (esq.) e DEEPFAKE (dir.) no mesmo frame.
- Calcula fluxo separadamente em cada metade para evitar "vazamento" no corte central.
- Mostra modos: original, overlay colorido do fluxo, vetores (quiver) e heatmap de magnitude.

Teclas:
  m: alterna modo de visualização
  espaço: pausa/continua
  s: salva snapshot
  +/-: ajusta limiar de detecção
  q: sai
"""

import cv2
import numpy as np
import os, time

# ========== CONFIG ==========
VIDEO_PATH = r"videos/Design sem nome.mp4"  # <-- ajuste aqui
DISPLAY_MAX_WIDTH = 1280                  # redimensiona p/ caber na tela (0 = não redimensiona)
FLOW_DOWNSCALE = 0.5                      # calcula fluxo numa escala menor p/ ganho de desempenho
ALPHA_OVERLAY = 0.55                      # transparência do overlay colorido
VECTOR_STEP = 16                          # espaçamento entre vetores (px) no modo "Vetores"
VECTOR_SCALE = 2.0                        # escala do tamanho dos vetores desenhados

# ========== MODOS ==========
MODES = ["Original", "Overlay", "Vetores", "Heatmap"]
mode_idx = 1  # começa em "Overlay"

def compute_flow(prev_bgr, curr_bgr):
    """Calcula fluxo Farnebäck + artefatos de visualização (mapa HSV e magnitude)."""
    # opcionalmente reduz para acelerar
    if FLOW_DOWNSCALE != 1.0:
        prev_bgr_small = cv2.resize(prev_bgr, None, fx=FLOW_DOWNSCALE, fy=FLOW_DOWNSCALE, interpolation=cv2.INTER_LINEAR)
        curr_bgr_small = cv2.resize(curr_bgr, None, fx=FLOW_DOWNSCALE, fy=FLOW_DOWNSCALE, interpolation=cv2.INTER_LINEAR)
    else:
        prev_bgr_small = prev_bgr
        curr_bgr_small = curr_bgr

    prev_g = cv2.cvtColor(prev_bgr_small, cv2.COLOR_BGR2GRAY)
    curr_g = cv2.cvtColor(curr_bgr_small, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_g, curr_g, None,
        pyr_scale=0.5, levels=3,
        winsize=25, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0
    )
    fx, fy = flow[...,0], flow[...,1]
    mag, ang = cv2.cartToPolar(fx, fy)

    # HSV: H=direção, S=255, V=mag normalizada (0..255)
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hsv = np.zeros((*ang.shape, 3), dtype=np.uint8)
    hsv[...,0] = (ang * 180 / np.pi / 2).astype(np.uint8)  # 0..2π -> 0..180
    hsv[...,1] = 255
    hsv[...,2] = mag_u8
    flow_bgr_small = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # reescala para o tamanho original do half-frame
    flow_bgr = cv2.resize(flow_bgr_small, (prev_bgr.shape[1], prev_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    mag_big  = cv2.resize(mag,            (prev_bgr.shape[1], prev_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

    return flow_bgr, mag_big, fx, fy, flow_bgr_small, mag  # retorna também versões small pra vetores

def draw_vectors(img_bgr, fx_small, fy_small, color=(255,255,255)):
    """Desenha um campo de vetores (setinhas) amostrando um grid."""
    h_s, w_s = fx_small.shape
    step = max(8, int(VECTOR_STEP * FLOW_DOWNSCALE))  # mantém densidade parecida após downscale
    for y in range(step//2, h_s, step):
        for x in range(step//2, w_s, step):
            dx = fx_small[y, x]
            dy = fy_small[y, x]
            # ponto origem na imagem grande
            X = int(x / FLOW_DOWNSCALE)
            Y = int(y / FLOW_DOWNSCALE)
            # destino (escala do vetor)
            X2 = int(X + VECTOR_SCALE * dx / FLOW_DOWNSCALE)
            Y2 = int(Y + VECTOR_SCALE * dy / FLOW_DOWNSCALE)
            cv2.arrowedLine(img_bgr, (X, Y), (X2, Y2), color, 1, tipLength=0.3)

def detect_face_swap_artifacts(frame):
    """Detecta artefatos específicos de face swap com foco na diferenciação"""
    # 1. Detector de face
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)

    if len(faces) == 0:
        return 0, 0, 0

    x, y, w, h = faces[0]  # Primeira face
    face_roi = frame[y:y+h, x:x+w]
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    # 1. DETECÇÃO DE BORDAS IRREGULARES (típico de face swap)
    # Face swaps deixam bordas cortantes e irregulares
    edges = cv2.Canny(gray_face, 50, 150)

    # Analisa densidade de bordas em regiões específicas (contorno do rosto)
    h_roi, w_roi = gray_face.shape
    edge_regions = [
        edges[:h_roi//4, :],           # topo (testa)
        edges[3*h_roi//4:, :],         # baixo (queixo)
        edges[:, :w_roi//6],           # esquerda (bochecha)
        edges[:, 5*w_roi//6:],         # direita (bochecha)
    ]

    edge_density = sum(np.mean(region) for region in edge_regions if region.size > 0)

    # 2. ANÁLISE DE TEXTURA ARTIFICIAL
    # Face swaps têm padrões de textura diferentes (mais suaves ou artificiais)
    # Usa análise de gradientes para detectar mudanças abruptas

    # Gradientes em X e Y
    grad_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitude dos gradientes
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Face swaps têm gradientes mais suaves (menos variação natural)
    texture_variance = np.var(gradient_mag)

    # 3. INCONSISTÊNCIA DE ILUMINAÇÃO
    # Face swaps frequentemente têm iluminação inconsistente
    # Divide o rosto em quadrantes e analisa diferenças de brilho
    lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0].astype(float)

    # Divide em 4 quadrantes
    h_mid, w_mid = h_roi//2, w_roi//2
    quadrants = [
        l_channel[:h_mid, :w_mid],      # superior esquerdo
        l_channel[:h_mid, w_mid:],      # superior direito
        l_channel[h_mid:, :w_mid],      # inferior esquerdo
        l_channel[h_mid:, w_mid:]       # inferior direito
    ]

    quadrant_means = [np.mean(q) for q in quadrants]
    lighting_variance = np.var(quadrant_means)

    return edge_density, texture_variance, lighting_variance

def put_label(img, text, org=(10, 28), color=(255,255,255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

def main():
    global mode_idx
    threshold = 3.0  # limiar inicial ajustável

    # Buffer temporal para estabilizar detecção
    score_buffer_L = []
    score_buffer_R = []
    buffer_size = 10

    if not os.path.exists(VIDEO_PATH):
        print(f"[ERRO] Arquivo não encontrado: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERRO] Não consegui abrir o vídeo.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms = 1

    ok, prev = cap.read()
    if not ok:
        print("[ERRO] Vídeo vazio.")
        return

    # redimensiona para tela se necessário
    if DISPLAY_MAX_WIDTH and prev.shape[1] > DISPLAY_MAX_WIDTH:
        scale = DISPLAY_MAX_WIDTH / prev.shape[1]
        prev = cv2.resize(prev, (DISPLAY_MAX_WIDTH, int(prev.shape[0]*scale)))

    win_name = "Fluxo Óptico — Esquerda: REAL | Direita: DEEPFAKE"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    paused = False
    frame_count = 0
    snap_idx = 0

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            if DISPLAY_MAX_WIDTH and frame.shape[1] > DISPLAY_MAX_WIDTH:
                scale = DISPLAY_MAX_WIDTH / frame.shape[1]
                frame = cv2.resize(frame, (DISPLAY_MAX_WIDTH, int(frame.shape[0]*scale)))

            H, W = frame.shape[:2]
            mid = W // 2

            # halves
            prevL, prevR = prev[:, :mid], prev[:, mid:]
            currL, currR = frame[:, :mid], frame[:, mid:]

            # calcula fluxo por metade
            flowL_bgr, magL, fxL_big, fyL_big, flowL_small, magL_small = compute_flow(prevL, currL)
            flowR_bgr, magR, fxR_big, fyR_big, flowR_small, magR_small = compute_flow(prevR, currR)

            # métricas simples por lado (usando float64 p/ evitar erros de tipo)
            mL = float(np.asarray(magL, dtype=np.float64).mean())
            sL = float(np.asarray(magL, dtype=np.float64).std())
            mR = float(np.asarray(magR, dtype=np.float64).mean())
            sR = float(np.asarray(magR, dtype=np.float64).std())

            # NOVA DETECÇÃO DE FACE SWAP
            edgeL, textL, lightL = detect_face_swap_artifacts(currL)
            edgeR, textR, lightR = detect_face_swap_artifacts(currR)

            # Score combinado (pesos ajustados para face swap)
            fakeL_score = (edgeL * 0.3 + textL * 0.0001 + lightL * 0.7)
            fakeR_score = (edgeR * 0.3 + textR * 0.0001 + lightR * 0.7)

            # Adiciona ao buffer temporal
            score_buffer_L.append(fakeL_score)
            score_buffer_R.append(fakeR_score)

            # Mantém tamanho do buffer
            if len(score_buffer_L) > buffer_size:
                score_buffer_L.pop(0)
                score_buffer_R.pop(0)

            # Médias temporais (mais estáveis)
            avg_L = np.mean(score_buffer_L)
            avg_R = np.mean(score_buffer_R)

            # Detecção baseada em DIFERENÇA RELATIVA (o fake deve ser consistentemente maior)
            score_ratio = avg_R / (avg_L + 0.1)  # evita divisão por zero

            # Lógica melhorada: fake é detectado quando há diferença consistente
            is_fake_L = (score_ratio < 0.7) and (avg_L > threshold)  # L muito maior que R
            is_fake_R = (score_ratio > 1.4) and (avg_R > threshold)  # R muito maior que L

            # Debug info a cada 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: AvgL={avg_L:.2f}, AvgR={avg_R:.2f}, Ratio={score_ratio:.2f}, T={threshold:.2f}")

            # prepara canvas de exibição conforme modo
            mode = MODES[mode_idx]
            if mode == "Original":
                left_vis  = currL.copy()
                right_vis = currR.copy()

            elif mode == "Overlay":
                left_vis  = cv2.addWeighted(currL, 1-ALPHA_OVERLAY, flowL_bgr, ALPHA_OVERLAY, 0)
                right_vis = cv2.addWeighted(currR, 1-ALPHA_OVERLAY, flowR_bgr, ALPHA_OVERLAY, 0)

            elif mode == "Vetores":
                left_vis  = currL.copy()
                right_vis = currR.copy()
                draw_vectors(left_vis,  flowL_small[...,0], flowL_small[...,1], color=(255,255,255))
                draw_vectors(right_vis, flowR_small[...,0], flowR_small[...,1], color=(255,255,255))

            elif mode == "Heatmap":
                # heatmap da magnitude: escala para 0..255 e aplica colormap
                magL_u8 = cv2.normalize(magL, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                magR_u8 = cv2.normalize(magR, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                left_vis  = cv2.applyColorMap(magL_u8, cv2.COLORMAP_TURBO)
                right_vis = cv2.applyColorMap(magR_u8, cv2.COLORMAP_TURBO)

            # concatena e rotula
            vis = np.hstack([left_vis, right_vis])

            # Labels com indicação de detecção
            real_color = (80, 240, 80) if not is_fake_L else (80, 80, 240)
            fake_color = (60, 140, 255) if not is_fake_R else (0, 0, 255)

            put_label(vis, f"REAL | avg={avg_L:.2f}", (10, 28), real_color)
            put_label(vis, f"{mode} | ratio={score_ratio:.2f}", (W//2 - 100, 28), (255, 255, 0))
            put_label(vis, f"DEEPFAKE | avg={avg_R:.2f}", (W//2 + 10, 28), fake_color)

            # Adiciona alerta visual para deepfake detectado
            if is_fake_R:
                cv2.rectangle(vis, (W//2, 0), (W, H), (0, 0, 255), 8)
                put_label(vis, "DEEPFAKE DETECTED!", (W//2 + 10, 70), (0, 0, 255))
                put_label(vis, f"Threshold: {threshold}", (W//2 + 10, 110), (255, 255, 255))

            if is_fake_L:
                cv2.rectangle(vis, (0, 0), (W//2, H), (0, 0, 255), 8)
                put_label(vis, "FAKE DETECTED!", (10, 70), (0, 0, 255))

            cv2.imshow(win_name, vis)
            prev = frame.copy()
            frame_count += 1

        key = cv2.waitKey(delay_ms if not paused else 30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mode_idx = (mode_idx + 1) % len(MODES)
        elif key == ord(' '):
            paused = not paused
        elif key == ord('s'):
            fname = f"snapshot_flow_{mode_idx}_{snap_idx:03d}.png"
            cv2.imwrite(fname, vis)
            print(f"[snap] salvo: {fname}")
            snap_idx += 1
        elif key == ord('+') or key == ord('='):
            threshold += 0.1
            print(f"[threshold] aumentado para: {threshold:.2f}")
        elif key == ord('-'):
            threshold = max(0.1, threshold - 0.1)
            print(f"[threshold] diminuído para: {threshold:.2f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()