# Compreensão da Arquitetura — Deep-Live-Cam

> Documento de análise de contexto do projeto **Deep-Live-Cam v2.1.6** (GitHub Edition).
> Objetivo: oferecer um mapa mental completo do código para quem precisa entender,
> manter ou estender o projeto.

---

## 1. Visão Geral

Deep-Live-Cam é uma aplicação de **troca de rosto (face swap) em tempo real** que cria
deepfakes a partir de **uma única imagem de origem**. Suporta:

- Face swap ao vivo via webcam
- Processamento de vídeos em lote
- Múltiplos rostos (trocar todos os rostos detectados)
- Mapeamento de rostos (associar rostos de origem diferentes a alvos específicos)
- Máscara de boca (preservar o movimento original da boca)
- Aceleração por hardware (CUDA, CoreML, DirectML)
- Pipeline de vídeo **em memória** (pipes FFmpeg, sem arquivos temporários)
- Filtro de conteúdo NSFW integrado (`opennsfw2`)

**Stack principal:** Python 3.9+, PySide6 (GUI Qt6), InsightFace (detecção/reconhecimento),
ONNX Runtime (inferência), FFmpeg (codificação de vídeo), OpenCV/NumPy.

---

## 2. Estrutura de Diretórios

```
Deep-Live-Cam/
├── run.py                       # Ponto de entrada
├── requirements.txt             # Dependências
├── benchmark_pipeline.py        # Benchmark de performance
├── locales/                     # Traduções (ru, id, etc.)
├── models/                      # Modelos ONNX (inswapper, GFPGAN)
├── tests/                       # Testes unitários (6 arquivos)
└── modules/
    ├── core.py                  # Orquestração principal do processamento
    ├── globals.py               # Estado/configuração global
    ├── ui.py                    # GUI PySide6 (~57KB)
    ├── face_analyser.py         # Detecção e reconhecimento de rostos
    ├── video_capture.py         # Captura de webcam (classe VideoCapturer)
    ├── capturer.py              # Extração de frames de vídeo
    ├── utilities.py             # Helpers FFmpeg, frames, temp dir
    ├── gpu_processing.py        # Operações de imagem aceleradas
    ├── cluster_analysis.py      # K-means para mapeamento de rostos
    ├── onnx_optimize.py         # Otimização de modelos ONNX (CoreML)
    ├── platform_info.py         # Detecção de hardware
    ├── metadata.py              # Nome/versão/edição
    └── processors/frame/
        ├── core.py              # Framework de processadores de frame
        ├── face_swapper.py      # Troca de rosto (módulo maior, ~72KB)
        ├── face_enhancer.py     # Aprimoramento GFPGAN
        ├── face_enhancer_gpen256.py / gpen512.py
        ├── face_masking.py      # Máscara de boca + blending
        └── _onnx_enhancer.py    # Gerenciamento de sessões ONNX
```

---

## 3. Fluxo de Inicialização

```
run.py
 ├─ platform_info.print_banner()      # Detecta CUDA/CoreML/DML/CPU
 └─ modules.core.run()
     ├─ parse_args()                  # Argumentos CLI (core.py)
     ├─ pre_check()                   # Python 3.9+, ffmpeg presente
     ├─ limit_resources()             # Limites de memória, GPU growth
     ├─ headless?  → start()          # Modo CLI / lote
     └─ senão      → ui.init().mainloop()   # GUI PySide6
```

**Prioridade de execution provider:** `cuda > rocm > coreml > dml > cpu`.

**Sugestão de threads:** DirectML/ROCM = 1 (inferência serial), CUDA = 2,
CPU = `max(4, cpu_count-2)` limitado a 16.

---

## 4. Pipeline de Processamento (Face Swap)

A arquitetura tem **duas vias**:

### A) Pipeline em memória (padrão, modo sem mapeamento)
`modules/processors/frame/core.py`

```
FFmpeg reader (rawvideo BGR24)
   └─> buffer de frame (numpy)
        ├─ detecção de rosto pipelined (thread em background — frame N+1 enquanto N codifica)
        ├─ face_swapper.process_frame()
        ├─ face_enhancer.process_frame()  (opcional)
        └─ face_masking (máscara de boca, opcional)
   └─> FFmpeg writer (h264_nvenc/libx264) → MP4/MKV + áudio restaurado
```

**Otimizações-chave:**
- Bytes BGR24 brutos fluem pelos pipes → elimina centenas de MB de I/O em disco
- Detecção sobreposta à codificação (pipelining em GPU)
- Fallback de encoder de hardware → software

### B) Fallback baseado em disco (necessário para mapeamento de rostos)
`modules/core.py`

1. Extrai frames para PNGs temporários
2. Processa em paralelo (`ThreadPoolExecutor`)
3. Recodifica a sequência PNG em vídeo
4. Restaura o áudio original

---

## 5. Módulos Centrais

### `modules/globals.py` — Estado global
Variáveis de configuração compartilhadas por todo o app:
- Caminhos: `source_path`, `target_path`, `output_path`
- Flags: `keep_fps`, `keep_audio`, `many_faces`, `map_faces`, `mouth_mask`, `nsfw_filter`
- Modo ao vivo: `live_mirror`, `webcam_preview_running`, `show_fps`
- Sistema: `max_memory`, `execution_providers`, `execution_threads`, `headless`
- Ajustes de swap: `opacity`, `sharpness`, `mask_feather_ratio`, `mouth_mask_size`
- Interpolação temporal: `enable_interpolation`, `interpolation_weight`
- Mapeamento: `source_target_map`, `simple_map`

### `modules/face_analyser.py` — Detecção e reconhecimento
- `get_face_analyser()`: singleton thread-safe do InsightFace (`buffalo_l`, det 640×640)
- `_analyse_faces(frame)`: detecção otimizada — só carrega `landmark_2d_106`
  se realmente necessário (máscara de boca ou enhancers ativos)
- Funções de mapeamento: extração de rostos únicos de imagem/vídeo via K-means
  (`get_unique_faces_from_target_video`, `simplify_maps`)

### `modules/processors/frame/face_swapper.py` — Troca de rosto (núcleo)
- Modelo: `inswapper_128_fp16.onnx` via ONNX Runtime
- Transformação afim para alinhar à pose/ângulo do alvo
- **Blending de Poisson** nas bordas (máscaras elípticas em cache)
- Correção de cor em espaço LAB (em `face_masking.py`)
- Interpolação de frames para suavização temporal

### `modules/processors/frame/core.py` — Framework de processadores
Interface que todo processador implementa:
`pre_check()`, `pre_start()`, `process_frame()`, `process_image()`, `process_video()`.
Funções: `get_frame_processors_modules()`, `multi_process_frame()`,
`process_video_in_memory()`.

### `modules/ui.py` — GUI PySide6
Classe `_Window(QMainWindow)`, tema escuro (QSS). Três modos:
1. Imagem/Vídeo (seleção de origem/alvo)
2. Webcam (preview ao vivo)
3. Mapeador de rostos (multi-face)

Threading: thread principal Qt + thread de preview (VideoCapturer) +
`ThreadPoolExecutor` para frames. Atualizações via sinais Qt.

### `modules/video_capture.py` — Captura de webcam
Classe `VideoCapturer` multiplataforma. Windows: DSHOW → MSMF → CAP_ANY
(via `pygrabber`). Mede FPS empiricamente; usa codec MJPG para frames USB comprimidos.

### `modules/utilities.py` — FFmpeg e frames
`run_ffmpeg()`, `detect_fps()`, `extract_frames()`, `create_video()`,
`restore_audio()`, gestão de diretório temporário.

### `modules/cluster_analysis.py` — Clustering de rostos
K-means (método do cotovelo) para identificar identidades únicas em vídeos
para o mapeamento de rostos.

---

## 6. Dependências Principais

| Pacote | Função |
|--------|--------|
| `numpy` (<2) | Arrays |
| `opencv-python` 4.10 | Visão computacional |
| `insightface` 0.7.3 | Detecção/reconhecimento (buffalo_l) |
| `onnx` / `onnxruntime-gpu` / `onnxruntime-silicon` | Inferência |
| `PySide6` (>=6.7) | GUI Qt6 |
| `tensorflow` (>=2.15) | GPU/GFPGAN |
| `opennsfw2` | Filtro NSFW |
| `pillow`, `tqdm`, `psutil` | I/O, progresso, monitoramento |

**Modelos externos** (baixados no 1º uso):
`inswapper_128_fp16.onnx` (swap), `GFPGANv1.4.onnx` (enhance), `buffalo_l` (detecção).
**Runtime:** `ffmpeg` + `ffprobe`.

---

## 7. Fluxos de Dados Resumidos

**Webcam ao vivo:**
```
VideoCapturer.read() → detecção rápida → face_swapper.process_frame()
   → (enhancer opcional) → exibe no widget (~30 fps)
```

**Vídeo em memória:**
```
pipe FFmpeg (BGR24) → reshape (H,W,3) → detecção pipelined
   → swap → enhance → máscara → pipe encoder → MP4 + áudio
```

**Mapeamento de rostos:**
```
extrai frames → detecta todos os rostos → coleta embeddings
   → K-means (identidades únicas) → monta source_target_map
   → na execução: cada rosto → centroide mais próximo → rosto de origem mapeado
```

---

## 8. Detalhes Notáveis de Implementação

1. **Detecção pipelined** — detecta o próximo frame em background enquanto o atual codifica
2. **Máscaras elípticas em cache** — reusa a máscara quando o rosto quase não se move
3. **Otimização ONNX para CoreML** (`onnx_optimize.py`) — folding de nós (21ms → 4ms no Apple Silicon)
4. **Lock DML** — serializa inferência no DirectML (não é thread-safe)
5. **Pipes FFmpeg em memória** — sem arquivos temporários no caminho rápido
6. **Carregamento preguiçoso de landmarks** — pula o modelo de 106 pontos quando só o swapper está ativo

**Casos de borda tratados:** sem rosto na origem (passa o frame original),
rosto fora do quadro (clampa coordenadas), troca de identidades no vídeo (clustering),
encoder de hardware indisponível (fallback para software).

---

## 9. Testes

Em `tests/` (6 arquivos):
- `test_cluster_analysis.py` — clustering K-means
- `test_face_analyser_get_one_face.py` — detecção de rosto
- `test_gpu_processing.py` — fallback das operações de GPU
- `test_multi_process_frame.py` — paralelização de frames
- `test_needs_landmark.py` — lógica de carregamento de landmarks
- `test_utilities_paths.py` — normalização de caminhos e temp dir

---

## 10. Resumo Mental

> **Detecção (InsightFace) → Troca (inswapper ONNX) → Aprimoramento (GFPGAN) → Blending (Poisson + LAB)**

O projeto separa claramente as responsabilidades e prioriza performance
(pipelines em memória, detecção pipelined, máscaras em cache, aceleração ONNX).
Funciona em Windows (DirectML/DSHOW), macOS (CoreML) e Linux (V4L2/CPU),
com GUI interativa (PySide6) e modo CLI para lote. Os módulos mais densos são
`face_swapper.py` (alinhamento/blending) e `ui.py` (interface).
