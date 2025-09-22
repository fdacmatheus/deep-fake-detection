# Detector de Deepfake - Face Swap Detection

Sistema de detecção de deepfakes em tempo real especializado em face swaps, desenvolvido para identificar vídeos onde rostos foram digitalmente substituídos.

## 📋 Visão Geral

Este projeto implementa um detector de deepfake focado especificamente em **face swaps** (troca de rostos), usando técnicas de visão computacional para identificar artefatos característicos deste tipo de manipulação.

### Características Principais

- ✅ **Detecção em tempo real** de face swaps
- ✅ **Análise temporal estabilizada** (buffer de frames)
- ✅ **Interface visual interativa** com múltiplos modos
- ✅ **Threshold ajustável** em tempo real
- ✅ **Algoritmo específico** para artefatos de face swap

## 🏗️ Arquitetura do Algoritmo

### 1. Detecção de Artefatos de Face Swap

O algoritmo foca em três características principais dos face swaps:

#### **Edge Density Analysis**
```python
# Detecta bordas irregulares onde o rosto foi "colado"
edges = cv2.Canny(gray_face, 50, 150)
edge_regions = [
    edges[:h_roi//4, :],           # topo (testa)
    edges[3*h_roi//4:, :],         # baixo (queixo)
    edges[:, :w_roi//6],           # esquerda (bochecha)
    edges[:, 5*w_roi//6:],         # direita (bochecha)
]
```

**Por que funciona**: Face swaps deixam bordas cortantes e irregulares nas regiões de transição entre o rosto substituído e o original.

#### **Texture Variance Analysis**
```python
# Detecta texturas artificiais típicas de GANs
grad_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
texture_variance = np.var(gradient_mag)
```

**Por que funciona**: GANs produzem texturas com padrões de gradiente diferentes dos rostos naturais.

#### **Lighting Inconsistency Analysis**
```python
# Detecta inconsistências de iluminação
quadrants = [
    l_channel[:h_mid, :w_mid],      # superior esquerdo
    l_channel[:h_mid, w_mid:],      # superior direito
    l_channel[h_mid:, :w_mid],      # inferior esquerdo
    l_channel[h_mid:, w_mid:]       # inferior direito
]
lighting_variance = np.var(quadrant_means)
```

**Por que funciona**: Face swaps frequentemente têm iluminação inconsistente entre diferentes regiões do rosto.

### 2. Estabilização Temporal

```python
# Buffer temporal para evitar detecções intermitentes
score_buffer_L.append(fakeL_score)
score_buffer_R.append(fakeR_score)

# Médias móveis para estabilizar
avg_L = np.mean(score_buffer_L)
avg_R = np.mean(score_buffer_R)
```

**Benefícios**:
- Elimina "piscadas" na detecção
- Reduz falsos positivos
- Torna a detecção mais confiável

### 3. Lógica de Comparação Relativa

```python
# Comparação entre lados
score_ratio = avg_R / (avg_L + 0.1)

# Detecção baseada em diferença significativa
is_fake_L = (score_ratio < 0.7) and (avg_L > threshold)
is_fake_R = (score_ratio > 1.4) and (avg_R > threshold)
```

**Vantagens**:
- Mais robusto que thresholds absolutos
- Adapta-se a diferentes condições de vídeo
- Foca na diferença entre real vs fake

## 🚀 Como Usar

### Pré-requisitos

```bash
pip install opencv-python numpy
```

### Execução

1. **Configure o vídeo** no arquivo:
   ```python
   VIDEO_PATH = r"videos/seu_video.mp4"
   ```

2. **Execute o programa**:
   ```bash
   python deepfakedetectorlimiar.py
   ```

### Controles Interativos

| Tecla | Função |
|-------|--------|
| `m` | Alterna modo de visualização |
| `espaço` | Pausa/continua reprodução |
| `s` | Salva screenshot |
| `+` | Aumenta threshold de detecção |
| `-` | Diminui threshold de detecção |
| `q` | Sair |

### Modos de Visualização

1. **Original**: Mostra o vídeo original
2. **Overlay**: Sobrepõe fluxo óptico colorido
3. **Vetores**: Mostra vetores de movimento
4. **Heatmap**: Visualiza magnitude do movimento

## 📊 Interpretação dos Resultados

### Interface Visual

- **Moldura Verde**: Lado identificado como real
- **Moldura Vermelha**: Lado identificado como fake
- **"DEEPFAKE DETECTED!"**: Alerta visual quando detectado

### Métricas Exibidas

```
REAL | avg=2.34        DEEPFAKE | avg=5.67
     ratio=2.41
```

- **avg**: Média temporal do score de detecção
- **ratio**: Proporção entre lado direito e esquerdo
- **ratio > 1.4**: Indica que o lado direito é fake
- **ratio < 0.7**: Indica que o lado esquerdo é fake

### Debug no Console

```
Frame 120: AvgL=2.34, AvgR=5.67, Ratio=2.41, T=3.00
```

- **AvgL/AvgR**: Médias dos scores
- **Ratio**: Proporção R/L
- **T**: Threshold atual

## ⚙️ Configuração e Ajustes

### Parâmetros Principais

```python
# Threshold inicial (ajustável com +/-)
threshold = 3.0

# Tamanho do buffer temporal
buffer_size = 10

# Pesos das métricas
fakeL_score = (edgeL * 0.3 + textL * 0.0001 + lightL * 0.7)
#               bordas        textura         iluminação
```

### Ajuste de Sensibilidade

- **Threshold baixo (1-2)**: Mais sensível, pode ter falsos positivos
- **Threshold médio (3-5)**: Balanceado (recomendado)
- **Threshold alto (6+)**: Menos sensível, pode perder detecções

## 🔧 Troubleshooting

### Problema: Não detecta o fake
**Solução**:
1. Diminua o threshold com `-`
2. Verifique se há face detectada
3. Aguarde o buffer se encher (10 frames)

### Problema: Muitos falsos positivos
**Solução**:
1. Aumente o threshold com `+`
2. Verifique a qualidade do vídeo
3. Considere ajustar os pesos das métricas

### Problema: Detecção instável
**Solução**:
1. Aumente o `buffer_size`
2. Verifique se há movimento excessivo
3. Use modo "Original" para verificar detecção facial

## 🧠 Como Funciona: Detalhes Técnicos

### Pipeline de Processamento

1. **Captura de Frame** → Redimensiona se necessário
2. **Divisão em Metades** → Esquerda (real) vs Direita (fake)
3. **Detecção Facial** → Localiza rosto em cada lado
4. **Análise de Artefatos** → Calcula métricas específicas
5. **Buffer Temporal** → Adiciona ao histórico
6. **Comparação Relativa** → Calcula ratio entre lados
7. **Decisão Final** → Determina se é fake
8. **Visualização** → Exibe resultado

### Otimizações Implementadas

- **Downscaling**: Processa fluxo óptico em resolução menor
- **ROI Facial**: Foca apenas na região do rosto
- **Buffer Circular**: Mantém memória constante
- **Normalização**: Evita overflow em cálculos

## 📈 Limitações e Melhorias Futuras

### Limitações Atuais
- Funciona melhor com face swaps de qualidade média/baixa
- Requer que o rosto esteja claramente visível
- Sensível a mudanças bruscas de iluminação

### Melhorias Propostas
- Implementar redes neurais pré-treinadas
- Adicionar detecção de múltiplas faces
- Melhorar robustez a variações de qualidade
- Implementar análise temporal mais sofisticada

## 📚 Referências Técnicas

- **Optical Flow**: Farnebäck method for motion analysis
- **Face Detection**: Haar Cascade Classifiers
- **Edge Detection**: Canny edge detector
- **Gradient Analysis**: Sobel operators
- **Color Space**: LAB color space for lighting analysis

---

**Desenvolvido por**: Sistema de Visão Computacional
**Versão**: 2.0 - Face Swap Specialized
**Última atualização**: 2025