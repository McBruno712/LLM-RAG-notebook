# LLM-RAG-notebook

## Sistema de Question Answering con RAG (Schengen Area)

Este proyecto implementa un sistema de Generación Aumentada por Recuperación (RAG) diseñado para responder preguntas en español sobre el Acuerdo de Schengen y la Unión Europea utilizando modelos de lenguaje de última generación y búsqueda semántica.

## 🚀 Arquitectura del Proyecto

El sistema se divide en cinco etapas principales:

1. **Extracción de Documentos:** Uso de wikipedia-api para recopilar información estructurada sobre el Espacio Schengen, Tratados de la UE y Fronteras exteriores.
2. **Procesamiento de Texto (Chunking):** Implementación de RecursiveCharacterTextSplitter para fragmentar documentos en segmentos de 500 caracteres, optimizando la retención de contexto.
3. **Recuperación de Información (Retrieval):**
   - **Embeddings:** Generados con el modelo multilingüe intfloat/multilingual-e5-large.
   - **Búsqueda:** Uso de NearestNeighbors (scikit-learn) comparando distancias Euclídeas y Similitud Coseno.
4. **Generación de Respuestas (LLM):** Inferencia con modelos de la familia Llama 3 (Meta) cuantizados a 4 bits (bitsandbytes) para eficiencia en memoria:
   - Llama 3.1 8B Instruct
   - Llama 3.2 3B Instruct
5. **Evaluación:** Uso de la métrica BERTScore para medir la similitud semántica entre las respuestas generadas y un conjunto de prueba (Gold Standard) de 12 preguntas manuales.

## 📊 Experimentos y Resultados

Se realizaron 4 experimentos variando el modelo generador y las técnicas de prompting:

| Exp | Modelo LLM   | Técnica de Recuperación | Prompting | F1 BERTScore |
| --- | ------------ | ----------------------- | --------- | ------------ |
| 1   | Llama 3.1 8B | KNN (Euclídea)          | Zero-shot | 0.830        |
| 2   | Llama 3.2 3B | KNN (Euclídea)          | Zero-shot | 0.859        |
| 3   | Llama 3.1 8B | Similitud Coseno        | Few-shot  | 0.854        |
| 4   | Llama 3.2 3B | Similitud Coseno        | Few-shot  | 0.844        |

## 🛠️ Tecnologías Utilizadas

- Python (Google Colab)
- Hugging Face (transformers, accelerate, evaluate)
- LangChain (Text Splitters)
- Sentence Transformers (Bi-Encoders)
- PyTorch (Cuantización 4-bit)
