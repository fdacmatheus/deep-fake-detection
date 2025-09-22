# -*- coding: utf-8 -*-
"""
Sistema de Detecção de Deepfakes em Vídeos - PBL 2 UNDB
Combina técnicas determinísticas e explicáveis de processamento de imagens
para detectar indícios de manipulação em vídeos informativos.

Técnicas implementadas:
- Detecção de bordas (Canny/Sobel)
- Análise espectral (FFT/DCT)
- Métricas de cor (HSV/RGB)
- Análise temporal (Fluxo óptico)
- Análise regional (olhos, boca, face)
- Detecção de artefatos (halos, inconsistências)

Autores: Baseado nos requisitos do PBL 2 - Visão Computacional UNDB
"""

import cv2
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

# ==================== CONFIGURAÇÕES ====================
class Config:
    """Configurações do sistema de detecção"""
    # Vídeo
    VIDEO_PATH = "videos/manipulated_sequences/Deepfakes/c23/videos/183_253.mp4"
    FRAME_SAMPLE_RATE = 10  # frames por segundo para análise
    MAX_FRAMES_ANALYZE = 100  # máximo de frames para análise completa

    # ROI e Detecção
    USE_FACE_DETECTION = True
    MANUAL_ROI_SELECTION = True

    # Limiares de Detecção (ajustáveis)
    EDGE_ANOMALY_THRESHOLD = 0.15  # 15% de diferença em bordas
    COLOR_INCONSISTENCY_THRESHOLD = 0.20  # 20% inconsistência de cor
    TEMPORAL_FLICKER_THRESHOLD = 0.25  # 25% de variação temporal
    FREQUENCY_ARTIFACT_THRESHOLD = 0.30  # 30% de artefatos no espectro

    # Visualização
    SHOW_INTERMEDIATE_RESULTS = True
    SAVE_REPORT = True
    OUTPUT_DIR = "deepfake_analysis_output"

# ==================== MÓDULO DE EXTRAÇÃO ====================
class FrameExtractor:
    """Extrai e prepara frames do vídeo para análise"""

    @staticmethod
    def extract_frames(video_path: str, sample_rate: int = 10, max_frames: int = 100) -> Tuple[List[np.ndarray], float]:
        """Extrai frames do vídeo em intervalos regulares"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Erro ao abrir vídeo: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(fps / sample_rate))

        frames = []
        frame_ids = []

        for i in range(0, frame_count, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                frame_ids.append(i)
                if len(frames) >= max_frames:
                    break

        cap.release()
        print(f"✓ Extraídos {len(frames)} frames do vídeo (FPS original: {fps:.1f})")
        return frames, fps

    @staticmethod
    def preprocess_frame(frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Pré-processa frame para diferentes análises"""
        processed = {
            'original': frame,
            'gray': cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            'hsv': cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
            'lab': cv2.cvtColor(frame, cv2.COLOR_BGR2LAB),
            'enhanced': cv2.convertScaleAbs(frame, alpha=1.2, beta=20)  # Realce de contraste
        }
        return processed

# ==================== MÓDULO DE DETECÇÃO DE ROI ====================
class ROIDetector:
    """Detecta e rastreia região de interesse (face)"""

    def __init__(self):
        self.face_cascade = None
        self.roi_history = []

    def detect_face_roi(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detecta face usando Haar Cascade ou métodos alternativos"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Tenta detecção por cor de pele primeiro
        roi = self._detect_skin_region(frame)
        if roi is not None:
            return roi

        # Fallback para seleção manual ou região central
        h, w = frame.shape[:2]
        return (w//4, h//5, w//2, 3*h//5)  # ROI central padrão

    def _detect_skin_region(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detecta região de pele usando segmentação HSV"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Faixas de cor de pele em HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 5000:  # Área mínima
                x, y, w, h = cv2.boundingRect(largest)
                return (x, y, w, h)
        return None

    def track_roi(self, frames: List[np.ndarray], initial_roi: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
        """Rastreia ROI através dos frames usando template matching"""
        rois = [initial_roi]
        x, y, w, h = initial_roi
        template = frames[0][y:y+h, x:x+w]

        for frame in frames[1:]:
            # Template matching
            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > 0.7:  # Threshold de confiança
                new_x, new_y = max_loc
                rois.append((new_x, new_y, w, h))
                # Atualiza template com média ponderada
                template = cv2.addWeighted(template, 0.7, frame[new_y:new_y+h, new_x:new_x+w], 0.3, 0)
            else:
                rois.append(initial_roi)  # Mantém ROI anterior se não encontrar

        return rois

# ==================== MÓDULOS DE ANÁLISE ====================

class EdgeAnalyzer:
    """Analisa bordas e detecta descontinuidades"""

    @staticmethod
    def analyze_edges(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Analisa bordas usando Canny e Sobel"""
        x, y, w, h = roi
        roi_img = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

        # Detecção de bordas
        canny_edges = cv2.Canny(gray, 50, 150)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)

        # Análise de descontinuidades
        edge_density = np.sum(canny_edges > 0) / canny_edges.size

        # Detecta bordas irregulares no contorno
        contour_mask = np.zeros_like(gray)
        cv2.rectangle(contour_mask, (5, 5), (w-5, h-5), 255, 2)
        contour_edges = cv2.bitwise_and(canny_edges, contour_mask)
        contour_irregularity = np.std(contour_edges[contour_mask > 0])

        return {
            'edge_density': edge_density,
            'contour_irregularity': contour_irregularity,
            'mean_gradient': np.mean(sobel_mag),
            'std_gradient': np.std(sobel_mag),
            'edges_canny': canny_edges,
            'edges_sobel': sobel_mag
        }

class FrequencyAnalyzer:
    """Análise no domínio da frequência (FFT/DCT)"""

    @staticmethod
    def analyze_frequency(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Analisa espectro de frequência para detectar artefatos"""
        x, y, w, h = roi
        roi_img = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

        # FFT 2D
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log(np.abs(f_shift) + 1)

        # DCT
        gray_float = np.float32(gray) / 255.0
        dct = cv2.dct(gray_float)
        dct_log = np.log(np.abs(dct) + 1e-6)

        # Análise de artefatos
        # Detecta padrões de blocagem (blockiness)
        block_size = 8
        blocks_h = h // block_size
        blocks_w = w // block_size
        block_artifacts = []

        for i in range(blocks_h):
            for j in range(blocks_w):
                block = dct[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                # Alta energia em frequências específicas indica compressão
                high_freq_energy = np.sum(np.abs(block[4:, 4:]))
                low_freq_energy = np.sum(np.abs(block[:4, :4]))
                if low_freq_energy > 0:
                    ratio = high_freq_energy / low_freq_energy
                    block_artifacts.append(ratio)

        blockiness_score = np.std(block_artifacts) if block_artifacts else 0

        # Detecta suavização excessiva
        high_freq_ratio = np.sum(magnitude[magnitude.shape[0]//3:, :]) / np.sum(magnitude)

        return {
            'blockiness_score': blockiness_score,
            'high_freq_ratio': high_freq_ratio,
            'fft_magnitude': magnitude,
            'dct_coefficients': dct_log,
            'compression_artifacts': len([b for b in block_artifacts if b > 0.5])
        }

class ColorAnalyzer:
    """Analisa consistência e anomalias de cor"""

    @staticmethod
    def analyze_color_consistency(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Analisa consistência de cor entre face e regiões adjacentes"""
        x, y, w, h = roi
        roi_img = frame[y:y+h, x:x+w]

        # Converte para diferentes espaços de cor
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi_img, cv2.COLOR_BGR2LAB)

        # Estatísticas de cor da ROI
        hsv_mean = np.mean(hsv.reshape(-1, 3), axis=0)
        hsv_std = np.std(hsv.reshape(-1, 3), axis=0)
        lab_mean = np.mean(lab.reshape(-1, 3), axis=0)
        lab_std = np.std(lab.reshape(-1, 3), axis=0)

        # Analisa regiões específicas
        regions = {
            'upper': roi_img[:h//3, :],  # Testa/olhos
            'middle': roi_img[h//3:2*h//3, :],  # Nariz/bochechas
            'lower': roi_img[2*h//3:, :]  # Boca/queixo
        }

        region_stats = {}
        for name, region in regions.items():
            region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            region_stats[name] = {
                'mean': np.mean(region_hsv.reshape(-1, 3), axis=0),
                'std': np.std(region_hsv.reshape(-1, 3), axis=0)
            }

        # Calcula inconsistências entre regiões
        inconsistency_score = 0
        for i, (name1, stats1) in enumerate(region_stats.items()):
            for name2, stats2 in list(region_stats.items())[i+1:]:
                diff = np.linalg.norm(stats1['mean'] - stats2['mean'])
                inconsistency_score += diff

        # Detecta halos usando filtro mediano
        median_filtered = cv2.medianBlur(roi_img, 7)
        residual = cv2.absdiff(roi_img, median_filtered)
        halo_score = np.mean(residual)

        return {
            'hsv_mean': hsv_mean,
            'hsv_std': hsv_std,
            'lab_mean': lab_mean,
            'lab_std': lab_std,
            'region_inconsistency': inconsistency_score,
            'halo_score': halo_score,
            'region_stats': region_stats
        }

class TemporalAnalyzer:
    """Analisa consistência temporal e fluxo óptico"""

    @staticmethod
    def analyze_temporal_consistency(frames: List[np.ndarray], rois: List[Tuple[int, int, int, int]]) -> Dict[str, Any]:
        """Analisa mudanças temporais e detecta flicker"""
        temporal_metrics = []
        optical_flows = []

        for i in range(len(frames) - 1):
            frame1, frame2 = frames[i], frames[i+1]
            roi1, roi2 = rois[i], rois[i+1]

            # Extrai ROIs
            x1, y1, w1, h1 = roi1
            x2, y2, w2, h2 = roi2
            roi_img1 = frame1[y1:y1+h1, x1:x1+w1]
            roi_img2 = frame2[y2:y2+h2, x2:x2+w2]

            # Redimensiona se necessário
            if roi_img1.shape != roi_img2.shape:
                roi_img2 = cv2.resize(roi_img2, (w1, h1))

            # Calcula fluxo óptico
            gray1 = cv2.cvtColor(roi_img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(roi_img2, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            # Magnitude e ângulo do fluxo
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Detecta inconsistências
            mean_flow = np.mean(mag)
            std_flow = np.std(mag)
            max_flow = np.max(mag)

            # Detecta flicker de cor
            color_diff = np.mean(np.abs(roi_img1.astype(float) - roi_img2.astype(float)))

            # Analisa regiões específicas (olhos, boca)
            h_third = h1 // 3
            eyes_flow = np.mean(mag[:h_third, :])
            mouth_flow = np.mean(mag[2*h_third:, :])

            temporal_metrics.append({
                'mean_flow': mean_flow,
                'std_flow': std_flow,
                'max_flow': max_flow,
                'color_flicker': color_diff,
                'eyes_motion': eyes_flow,
                'mouth_motion': mouth_flow
            })

            optical_flows.append(mag)

        # Calcula estatísticas temporais globais
        all_means = [m['mean_flow'] for m in temporal_metrics]
        all_flickers = [m['color_flicker'] for m in temporal_metrics]

        return {
            'frame_metrics': temporal_metrics,
            'temporal_consistency': 1.0 - (np.std(all_means) / (np.mean(all_means) + 1e-6)),
            'flicker_score': np.mean(all_flickers),
            'motion_irregularity': np.std([m['eyes_motion'] - m['mouth_motion'] for m in temporal_metrics]),
            'optical_flows': optical_flows
        }

# ==================== MOTOR DE DECISÃO ====================
class DeepfakeDetector:
    """Sistema principal de detecção de deepfakes"""

    def __init__(self, config: Config):
        self.config = config
        self.frame_extractor = FrameExtractor()
        self.roi_detector = ROIDetector()
        self.edge_analyzer = EdgeAnalyzer()
        self.freq_analyzer = FrequencyAnalyzer()
        self.color_analyzer = ColorAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer()
        self.analysis_results = {}

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Executa análise completa do vídeo"""
        print("\n" + "="*60)
        print("🔍 SISTEMA DE DETECÇÃO DE DEEPFAKES - PBL 2 UNDB")
        print("="*60)

        # 1. Extração de frames
        print("\n[1/6] Extraindo frames do vídeo...")
        frames, fps = self.frame_extractor.extract_frames(
            video_path,
            self.config.FRAME_SAMPLE_RATE,
            self.config.MAX_FRAMES_ANALYZE
        )

        # 2. Detecção de ROI
        print("\n[2/6] Detectando região de interesse (face)...")
        initial_roi = self._get_roi(frames[0])
        rois = self.roi_detector.track_roi(frames, initial_roi)
        print(f"✓ ROI rastreada em {len(rois)} frames")

        # 3. Análise de bordas
        print("\n[3/6] Analisando bordas e descontinuidades...")
        edge_results = []
        for frame, roi in zip(frames[:10], rois[:10]):  # Amostra
            edge_results.append(self.edge_analyzer.analyze_edges(frame, roi))

        avg_edge_density = np.mean([r['edge_density'] for r in edge_results])
        avg_contour_irreg = np.mean([r['contour_irregularity'] for r in edge_results])
        print(f"✓ Densidade média de bordas: {avg_edge_density:.3f}")
        print(f"✓ Irregularidade de contorno: {avg_contour_irreg:.3f}")

        # 4. Análise de frequência
        print("\n[4/6] Analisando espectro de frequência...")
        freq_results = []
        for frame, roi in zip(frames[:10], rois[:10]):
            freq_results.append(self.freq_analyzer.analyze_frequency(frame, roi))

        avg_blockiness = np.mean([r['blockiness_score'] for r in freq_results])
        avg_compression = np.mean([r['compression_artifacts'] for r in freq_results])
        print(f"✓ Score de blocagem: {avg_blockiness:.3f}")
        print(f"✓ Artefatos de compressão: {avg_compression:.1f}")

        # 5. Análise de cor
        print("\n[5/6] Analisando consistência de cor...")
        color_results = []
        for frame, roi in zip(frames[:10], rois[:10]):
            color_results.append(self.color_analyzer.analyze_color_consistency(frame, roi))

        avg_inconsistency = np.mean([r['region_inconsistency'] for r in color_results])
        avg_halo = np.mean([r['halo_score'] for r in color_results])
        print(f"✓ Inconsistência regional: {avg_inconsistency:.3f}")
        print(f"✓ Score de halo: {avg_halo:.3f}")

        # 6. Análise temporal
        print("\n[6/6] Analisando consistência temporal...")
        temporal_results = self.temporal_analyzer.analyze_temporal_consistency(frames, rois)

        temporal_consistency = temporal_results['temporal_consistency']
        flicker_score = temporal_results['flicker_score']
        motion_irreg = temporal_results['motion_irregularity']
        print(f"✓ Consistência temporal: {temporal_consistency:.3f}")
        print(f"✓ Score de flicker: {flicker_score:.3f}")
        print(f"✓ Irregularidade de movimento: {motion_irreg:.3f}")

        # Compilar resultados
        self.analysis_results = {
            'video_info': {
                'path': video_path,
                'fps': fps,
                'frames_analyzed': len(frames)
            },
            'edge_analysis': {
                'density': avg_edge_density,
                'contour_irregularity': avg_contour_irreg,
                'anomaly_detected': avg_contour_irreg > self.config.EDGE_ANOMALY_THRESHOLD
            },
            'frequency_analysis': {
                'blockiness': avg_blockiness,
                'compression_artifacts': avg_compression,
                'anomaly_detected': avg_blockiness > self.config.FREQUENCY_ARTIFACT_THRESHOLD
            },
            'color_analysis': {
                'region_inconsistency': avg_inconsistency,
                'halo_score': avg_halo,
                'anomaly_detected': avg_inconsistency > self.config.COLOR_INCONSISTENCY_THRESHOLD
            },
            'temporal_analysis': {
                'consistency': temporal_consistency,
                'flicker': flicker_score,
                'motion_irregularity': motion_irreg,
                'anomaly_detected': flicker_score > self.config.TEMPORAL_FLICKER_THRESHOLD
            }
        }

        # Decisão final
        self._make_decision()

        # Visualização
        if self.config.SHOW_INTERMEDIATE_RESULTS:
            self._visualize_results(frames[0], rois[0], edge_results[0], freq_results[0])

        # Salvar relatório
        if self.config.SAVE_REPORT:
            self._save_report()

        return self.analysis_results

    def _get_roi(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Obtém ROI através de detecção automática ou seleção manual"""
        if self.config.MANUAL_ROI_SELECTION:
            print("📌 Selecione a região do rosto na janela...")
            roi = cv2.selectROI("Selecione o rosto", frame, showCrosshair=True)
            cv2.destroyWindow("Selecione o rosto")
            x, y, w, h = map(int, roi)
            if w > 0 and h > 0:
                return (x, y, w, h)

        # Tenta detecção automática
        roi = self.roi_detector.detect_face_roi(frame)
        if roi:
            return roi

        # ROI padrão (centro)
        h, w = frame.shape[:2]
        return (w//4, h//4, w//2, h//2)

    def _make_decision(self):
        """Toma decisão final baseada em múltiplos indicadores"""
        anomalies = []
        confidence_scores = []

        # Coleta anomalias detectadas
        for category in ['edge_analysis', 'frequency_analysis', 'color_analysis', 'temporal_analysis']:
            if self.analysis_results[category]['anomaly_detected']:
                anomalies.append(category.replace('_', ' ').title())

                # Calcula score de confiança baseado na severidade
                if category == 'edge_analysis':
                    score = min(1.0, self.analysis_results[category]['contour_irregularity'] / 0.3)
                elif category == 'frequency_analysis':
                    score = min(1.0, self.analysis_results[category]['blockiness'] / 0.5)
                elif category == 'color_analysis':
                    score = min(1.0, self.analysis_results[category]['region_inconsistency'] / 50)
                else:  # temporal
                    score = min(1.0, self.analysis_results[category]['flicker'] / 30)

                confidence_scores.append(score)

        # Decisão baseada em múltiplos indicadores
        num_anomalies = len(anomalies)
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0

        if num_anomalies >= 3 or (num_anomalies >= 2 and avg_confidence > 0.7):
            verdict = "DEEPFAKE DETECTADO"
            confidence = "ALTA" if avg_confidence > 0.8 else "MÉDIA"
        elif num_anomalies >= 2:
            verdict = "SUSPEITO"
            confidence = "MÉDIA"
        elif num_anomalies == 1 and avg_confidence > 0.6:
            verdict = "POSSÍVEL DEEPFAKE"
            confidence = "BAIXA"
        else:
            verdict = "AUTÊNTICO"
            confidence = "ALTA"

        self.analysis_results['decision'] = {
            'verdict': verdict,
            'confidence': confidence,
            'anomalies_found': anomalies,
            'num_indicators': num_anomalies,
            'confidence_score': avg_confidence
        }

        # Imprime resultado
        print("\n" + "="*60)
        print("📊 RESULTADO DA ANÁLISE")
        print("="*60)
        print(f"🎯 Veredito: {verdict}")
        print(f"📈 Confiança: {confidence} ({avg_confidence:.1%})")

        if anomalies:
            print(f"⚠️  Anomalias detectadas ({num_anomalies}):")
            for anomaly in anomalies:
                print(f"   • {anomaly}")

        print("\n💡 Explicação:")
        if num_anomalies >= 3:
            print("   Múltiplos indicadores apontam para manipulação do vídeo.")
            print("   Foram detectadas inconsistências significativas em várias")
            print("   dimensões da análise (espacial, temporal e espectral).")
        elif num_anomalies >= 2:
            print("   Alguns indicadores sugerem possível manipulação.")
            print("   Recomenda-se análise adicional por especialista.")
        elif num_anomalies == 1:
            print("   Detectada anomalia isolada que pode indicar manipulação")
            print("   ou ser resultado de compressão/processamento do vídeo.")
        else:
            print("   Não foram detectados indícios significativos de manipulação.")
            print("   O vídeo aparenta ser autêntico.")

    def _visualize_results(self, frame: np.ndarray, roi: Tuple[int, int, int, int],
                          edge_result: Dict, freq_result: Dict):
        """Visualiza resultados das análises"""
        x, y, w, h = roi
        roi_img = frame[y:y+h, x:x+w]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Análise de Deepfake - Resultados Visuais', fontsize=16)

        # Original com ROI
        frame_with_roi = frame.copy()
        cv2.rectangle(frame_with_roi, (x, y), (x+w, y+h), (0, 255, 0), 2)
        axes[0, 0].imshow(cv2.cvtColor(frame_with_roi, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Frame Original com ROI')
        axes[0, 0].axis('off')

        # ROI ampliada
        axes[0, 1].imshow(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Região de Interesse (Face)')
        axes[0, 1].axis('off')

        # Bordas Canny
        axes[0, 2].imshow(edge_result['edges_canny'], cmap='gray')
        axes[0, 2].set_title('Detecção de Bordas (Canny)')
        axes[0, 2].axis('off')

        # Magnitude FFT
        axes[1, 0].imshow(freq_result['fft_magnitude'], cmap='hot')
        axes[1, 0].set_title('Espectro FFT (Magnitude)')
        axes[1, 0].axis('off')

        # DCT
        axes[1, 1].imshow(freq_result['dct_coefficients'], cmap='viridis')
        axes[1, 1].set_title('Coeficientes DCT')
        axes[1, 1].axis('off')

        # Gradiente Sobel
        axes[1, 2].imshow(edge_result['edges_sobel'], cmap='gray')
        axes[1, 2].set_title('Gradiente Sobel')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()

    def _save_report(self):
        """Salva relatório detalhado em JSON"""
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.config.OUTPUT_DIR, f"deepfake_report_{timestamp}.json")

        # Converte arrays numpy para listas
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        report_data = convert_numpy(self.analysis_results)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"\n📄 Relatório salvo em: {report_path}")

# ==================== FUNÇÃO PRINCIPAL ====================
def main():
    """Função principal do sistema"""
    config = Config()

    # Verifica se o vídeo existe
    if len(sys.argv) > 1:
        config.VIDEO_PATH = sys.argv[1]

    if not os.path.exists(config.VIDEO_PATH):
        print(f"❌ Erro: Vídeo não encontrado: {config.VIDEO_PATH}")
        print("Uso: python deepfake_detector_pbl.py [caminho_do_video]")
        sys.exit(1)

    # Inicializa detector
    detector = DeepfakeDetector(config)

    try:
        # Executa análise
        results = detector.analyze_video(config.VIDEO_PATH)

        print("\n" + "="*60)
        print("✅ Análise concluída com sucesso!")
        print("="*60)

        # Instruções para o usuário
        print("\n📌 Próximos passos:")
        print("1. Revise o relatório JSON gerado para detalhes técnicos")
        print("2. Analise as visualizações para entender os indicadores")
        print("3. Compare com vídeos conhecidos como autênticos")
        print("4. Em caso de dúvida, consulte especialista forense")

    except Exception as e:
        print(f"\n❌ Erro durante análise: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()