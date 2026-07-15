# Representaciones vectoriales de texto

Comparación teórica y empírica de **TF-IDF**, **Latent Semantic Analysis (LSA)**, **Latent Dirichlet Allocation (LDA)** y **Skip-gram with Negative Sampling (SG-NS)** para la representación y clasificación de documentos textuales.

El proyecto combina el desarrollo matemático de cada técnica con implementaciones propias, versiones optimizadas mediante librerías de alto nivel y experimentos reproducibles sobre dos corpus de características muy diferentes: **Sentiment140** y **arXiv**.

---

## Descripción

Los algoritmos de aprendizaje automático necesitan entradas numéricas. Por tanto, antes de clasificar, agrupar o recuperar documentos, es necesario transformar el texto en vectores que conserven la información relevante del corpus.

Este proyecto analiza cuatro familias de representaciones:

- **TF-IDF:** representación estadística, dispersa y directamente interpretable.
- **LSA:** representación algebraica obtenida mediante reducción dimensional con SVD.
- **LDA:** representación probabilística de los documentos como mezclas de tópicos.
- **SG-NS:** embeddings neuronales de palabras agregados para construir vectores documentales densos.

La comparación no se limita al rendimiento predictivo. También se estudian:

- dimensionalidad;
- dispersión;
- memoria;
- tiempo de construcción;
- tiempo de entrenamiento del clasificador;
- geometría del espacio vectorial;
- interpretabilidad;
- sensibilidad al dominio del corpus.

---

## Objetivos

El objetivo general es analizar, formalizar y comparar distintas técnicas de representación vectorial de textos desde una doble perspectiva: **teórica** y **experimental**.

Los objetivos específicos son:

1. Formalizar matemáticamente TF-IDF, LSA, LDA y SG-NS.
2. Implementar manualmente las operaciones principales de cada método.
3. Comparar las implementaciones propias con versiones optimizadas de `scikit-learn`.
4. Evaluar las representaciones mediante un clasificador común.
5. Analizar el efecto del tipo de corpus sobre el rendimiento.
6. Estudiar el compromiso entre precisión, coste, dimensionalidad e interpretabilidad.
7. Comparar embeddings generales con embeddings entrenados en el dominio de aplicación.

---

## Técnicas analizadas

| Técnica | Enfoque | Representación resultante | Principal ventaja | Principal limitación |
|---|---|---|---|---|
| TF-IDF | Estadístico | Vector disperso de dimensión igual al vocabulario | Simplicidad e interpretación directa | Alta dimensionalidad y dispersión |
| LSA | Algebraico | Vector denso en un espacio latente | Reduce dimensionalidad y captura coocurrencias | Componentes menos interpretables |
| LDA | Probabilístico | Distribución de tópicos por documento | Representación compacta e interpretable | Coste elevado y dependencia de una estructura temática clara |
| SG-NS | Neuronal | Embeddings densos de palabras y documentos | Captura relaciones distribucionales | Depende del corpus de entrenamiento y pierde orden al agregar palabras |

---

## Fundamentos matemáticos

### TF-IDF

Para un término $t$, un documento $d$ y un corpus $D$:

$$
tfidf(t,d) = \operatorname{tf}(t,d) \cdot \operatorname{idf}(t,D)
$$

con

$$
\operatorname{tf}(t,d)
=
\frac{f(t,d)}
{\max_{t' \in d} f(t',d)}
$$

e

$$
\operatorname{idf}(t,D)
=
\log
\left(
\frac{|D|}
{|\{d \in D : t \in d\}|}
\right).
$$

### LSA

LSA parte de una matriz término-documento $A$ y aplica la descomposición en valores singulares:

$$
A = U\Sigma V^\top.
$$

La representación reducida conserva las $k$ componentes asociadas a los valores singulares más importantes:

$$
A_k = U_k\Sigma_kV_k^\top.
$$

Esta aproximación reduce el ruido y proyecta los documentos en un espacio semántico de menor dimensión.

### LDA

LDA representa cada documento mediante una distribución sobre $K$ tópicos:

$$
\theta_d =
(\theta_{d,1},\ldots,\theta_{d,K}),
$$

donde $\theta_{d,k}$ es el peso del tópico $k$ en el documento $d$.

La implementación manual utiliza **muestreo de Gibbs colapsado** para estimar las asignaciones de tópicos y las distribuciones documento-tópico y tópico-palabra.

### SG-NS

Skip-gram aprende embeddings de palabras prediciendo términos de contexto a partir de una palabra central. El muestreo negativo reemplaza el cálculo completo de `softmax` por una tarea de clasificación entre pares positivos y negativos.

Para construir un vector documental se emplea una media ponderada por TF-IDF:

$$
v_D =
\frac{
\sum_{w \in D}
\operatorname{tfidf}(w,D)\,v_w
}{
\sum_{w \in D}
\operatorname{tfidf}(w,D)
}.
$$

---

## Conjuntos de datos

### Sentiment140

Corpus de mensajes breves de Twitter orientado al análisis de sentimiento.

- Dataset original: **1.600.000 tuits**.
- Clases: sentimiento positivo y negativo.
- Muestra experimental balanceada: **20.000 documentos**.
- Entrenamiento: **16.000 documentos**.
- Evaluación: **4.000 documentos**.
- Características: textos breves, informales, ruidosos y con poco contexto.

### arXiv

Instantánea de metadatos de artículos científicos.

- Fuente textual: título y resumen.
- Problema binario:
  - `cs`: categoría principal perteneciente a Computer Science.
  - `no_cs`: resto de áreas.
- Muestra balanceada: **100.000 artículos**.
- Características: documentos más largos, técnicos, formales y semánticamente densos.

La combinación de ambos corpus permite comparar las representaciones en dos escenarios opuestos: lenguaje social breve frente a documentación científica especializada.

---

## Preprocesamiento

Se aplica un procedimiento común para que las diferencias observadas dependan de la representación y no de tratamientos textuales distintos.

El flujo incluye:

1. conversión a minúsculas;
2. eliminación de URL, etiquetas HTML y menciones;
3. normalización de espacios;
4. tokenización con spaCy;
5. eliminación de puntuación y números;
6. eliminación de stopwords;
7. conservación explícita de negaciones;
8. lematización;
9. eliminación de documentos vacíos;
10. comprobación de valores nulos y duplicados.

---

## Metodología experimental

Todas las representaciones se evalúan bajo un protocolo común:

1. preprocesamiento homogéneo;
2. muestra balanceada;
3. partición estratificada de entrenamiento y prueba;
4. ajuste de la representación sobre entrenamiento;
5. transformación del conjunto de prueba;
6. clasificación mediante **Random Forest**;
7. cálculo de métricas predictivas y computacionales;
8. análisis de interpretabilidad y geometría.

Se utiliza el mismo clasificador para aislar el efecto de la representación vectorial.

### Métricas predictivas

- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC

### Métricas computacionales y geométricas

- tiempo de construcción;
- tiempo de transformación;
- tiempo de entrenamiento y predicción;
- dimensionalidad;
- sparsity;
- memoria;
- norma media;
- similitud coseno intra-clase;
- similitud coseno inter-clase;
- distancia entre centroides;
- entropía de las distribuciones de tópicos.

---

## Resultados principales

> Los tiempos corresponden al entorno experimental utilizado y no deben interpretarse como benchmarks universales.

### Sentiment140

| Representación | Dim. | Accuracy | F1 | AUC-ROC | Tiempo representación (s) | Tiempo clasificador (s) | Sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF manual | 2435 | 0.7250 | 0.7250 | 0.7949 | 0.92 | 149.02 | 0.9980 |
| LSA manual | 100 | 0.7163 | 0.7162 | 0.7797 | 51.46 | 22.06 | 0.0144 |
| LDA manual | 10 | 0.5285 | 0.5280 | 0.5335 | 115.54 | 1.95 | 0.0000 |
| SG-NS Wikipedia + TF-IDF | 100 | 0.6148 | 0.6146 | 0.6681 | 332.28 | 44.83 | 0.0000 |
| SG-NS Sentiment140 + TF-IDF | 100 | 0.6968 | 0.6965 | 0.7624 | 31.85 | 19.89 | 0.0000 |

**Conclusiones sobre Sentiment140:**

- TF-IDF obtiene el mejor rendimiento global.
- LSA conserva un rendimiento cercano con una representación mucho más compacta.
- LDA presenta dificultades por la brevedad y el ruido de los tuits.
- SG-NS mejora cuando los embeddings se entrenan sobre el propio dominio.
- Las señales léxicas directas resultan especialmente importantes en clasificación de sentimiento.

### arXiv

| Representación | Dim. | Accuracy | F1 | AUC-ROC | Tiempo representación (s) | Tiempo clasificador (s) | Sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF manual | 5000 | 0.9200 | 0.9199 | 0.9692 | 1.97 | 49.22 | 0.9878 |
| LSA manual | 100 | 0.9193 | 0.9192 | 0.9681 | 157.27 | 15.63 | 0.0000 |
| LDA manual | 10 | 0.9118 | 0.9117 | 0.9647 | 1923.32 | 3.06 | 0.0000 |
| SG-NS Wikipedia + TF-IDF | 100 | 0.8785 | 0.8785 | 0.9444 | 332.28 | 28.10 | 0.0000 |
| SG-NS arXiv + TF-IDF | 100 | 0.9255 | 0.9255 | 0.9730 | 220.15 | 20.41 | 0.0000 |

**Conclusiones sobre arXiv:**

- TF-IDF, LSA y LDA obtienen resultados competitivos.
- LSA mantiene un rendimiento similar a TF-IDF con mucha menor dimensionalidad.
- LDA mejora respecto a Sentiment140 porque los documentos contienen una estructura temática más clara.
- SG-NS entrenado sobre arXiv obtiene el mejor resultado global.
- Los embeddings especializados superan a los embeddings generales de Wikipedia.

---

## Interpretabilidad

El proyecto incluye diferentes estrategias de análisis:

### TF-IDF

- términos con mayor IDF;
- términos relevantes por clase;
- similitud coseno entre términos;
- proyecciones PCA;
- recuperación de documentos similares.

### LSA

- valores singulares;
- energía acumulada;
- términos con mayor peso por componente;
- centroides de clase;
- proyecciones bidimensionales;
- reconstrucción aproximada del espacio de términos.

### LDA

- palabras principales por tópico;
- distribución de tópicos por documento;
- tópico dominante;
- diferencias de peso entre clases;
- entropía temática;
- matrices documento-tópico y tópico-palabra.

### SG-NS

- palabras más cercanas;
- similitud coseno;
- proyecciones locales con PCA;
- comparación entre embeddings generales y específicos;
- construcción y recuperación de documentos mediante SG-NS + TF-IDF.

---

## Implementaciones manuales y librerías

El repositorio incluye implementaciones propias de los métodos principales y comparaciones con versiones optimizadas.

| Método | Implementación manual | Implementación de referencia |
|---|---|---|
| TF-IDF | Cálculo explícito de TF, IDF y matriz documental | `TfidfVectorizer` |
| LSA | TF-IDF manual + SVD con NumPy | `TruncatedSVD` |
| LDA | Muestreo de Gibbs colapsado | `LatentDirichletAllocation` |
| SG-NS | Generación de pares, negative sampling y entrenamiento | TensorFlow / embeddings preentrenados |

Las versiones manuales priorizan la correspondencia con la formulación matemática. Las implementaciones de librería son más eficientes gracias al uso de matrices dispersas, operaciones vectorizadas y algoritmos optimizados.

---

## Tecnologías

- Python
- NumPy
- pandas
- SciPy
- scikit-learn
- spaCy
- TensorFlow
- Matplotlib
- tqdm
- Google Colab
- Google Drive

Los experimentos se ejecutaron principalmente en Google Colab. Cuando estuvo disponible, se utilizó una **NVIDIA Tesla T4 de 16 GB**, aunque varias etapas —como el preprocesamiento, la construcción del vocabulario y el cálculo de frecuencias— dependen principalmente de CPU y memoria.

---

## Instalación

```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_REPOSITORIO>

python -m venv .venv
source .venv/bin/activate
```

En Windows:

```powershell
.venv\Scripts\activate
```

Instalación de las dependencias principales:

```bash
pip install numpy pandas scipy scikit-learn spacy tensorflow matplotlib tqdm adjustText
python -m spacy download en_core_web_sm
```

---

## Ejecución

El flujo recomendado es:

1. descargar o preparar los datasets;
2. adaptar las rutas de entrada y salida;
3. ejecutar el preprocesamiento;
4. generar las representaciones;
5. entrenar el clasificador;
6. evaluar las métricas;
7. ejecutar los análisis de interpretabilidad.

Notebooks principales desarrollados durante el proyecto:

```text
Copia_de_sentiment_140_preprocess.ipynb
Sentiment140_vectors.ipynb
arxiv_representation.ipynb
```

Al utilizar Google Colab, revisa las rutas configuradas con el patrón:

```python
/content/drive/MyDrive/TFM/
```

---

## Reproducibilidad

Para favorecer comparaciones homogéneas:

- se utilizan semillas aleatorias fijas;
- las muestras se balancean por clase;
- la división de entrenamiento y prueba es estratificada;
- se reutiliza el mismo clasificador;
- se mantienen las mismas particiones al comparar representaciones;
- los tiempos de representación y clasificación se registran por separado.

Los resultados pueden variar según:

- hardware;
- versión de las librerías;
- tamaño del corpus;
- disponibilidad de GPU;
- dimensionalidad;
- hiperparámetros;
- estado de la instantánea de arXiv.

---

## Principales conclusiones

1. No existe una representación universalmente superior.
2. TF-IDF es una referencia muy competitiva para textos breves y señales léxicas directas.
3. LSA ofrece un buen equilibrio entre rendimiento y reducción dimensional.
4. LDA resulta más útil cuando los documentos tienen suficiente longitud y coherencia temática.
5. SG-NS depende fuertemente del dominio usado para entrenar los embeddings.
6. Los embeddings especializados pueden superar a los entrenados sobre corpus generales.
7. La eficiencia debe evaluarse sobre el pipeline completo, no solo sobre la construcción de la representación.
8. La interpretabilidad adopta formas diferentes en cada método.
9. Las implementaciones manuales facilitan la comprensión, pero no igualan la eficiencia de librerías optimizadas.

---

## Limitaciones

- La evaluación se centra en Sentiment140 y arXiv.
- Las tareas se simplifican a clasificación binaria.
- Se utiliza un único clasificador supervisado.
- Las implementaciones manuales priorizan claridad frente a eficiencia.
- La agregación de embeddings pierde información sobre el orden de las palabras.
- Las proyecciones PCA en dos dimensiones son exploratorias y no representan toda la estructura del espacio original.

---

## Líneas futuras

- evaluación en otros dominios e idiomas;
- clasificación multiclase y multietiqueta;
- recuperación de información y clustering;
- comparación con RNN, LSTM y GRU;
- incorporación de BERT y arquitecturas Transformer;
- embeddings contextuales;
- búsqueda sistemática de hiperparámetros;
- comparación con otros clasificadores;
- estudio de representaciones que preserven la estructura secuencial.

---

## Referencias

El marco teórico se apoya, entre otros, en los trabajos fundacionales de:

- Salton et al. sobre el modelo vectorial y TF-IDF.
- Deerwester et al. sobre Latent Semantic Analysis.
- Blei, Ng y Jordan sobre Latent Dirichlet Allocation.
- Mikolov et al. sobre Skip-gram y Negative Sampling.
- Go, Bhayani y Huang sobre Sentiment140.

La bibliografía completa se encuentra en el documento académico asociado al proyecto.

---

## Uso académico

Este repositorio acompaña un trabajo académico centrado en la relación entre:

- formulación matemática;
- implementación computacional;
- rendimiento predictivo;
- coste;
- dimensionalidad;
- geometría;
- interpretabilidad.

Al reutilizar código, resultados o figuras, se recomienda citar el trabajo y las fuentes originales de los datasets y algoritmos.
