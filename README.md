
### Sprache / Idioma:
[🇩🇪 Deutsch](#deutsch) | [🇪🇸 Español](#espa%C3%B1ol) | [🇬🇧 OMG](#omg) 

---
## Deutsch

### OpenNN vs. MLPack – Vergleich neuronaler Netzwerkbibliotheken in C++
Dieses Repository dokumentiert einen praktischen Vergleich zwischen zwei populären C++-Bibliotheken für maschinelles Lernen: [OpenNN](https://www.opennn.net/) und [MLPack](https://www.mlpack.org/), mit Fokus auf neuronale Netzwerke (Multilayer Perceptrons).

### Zielsetzung
Der Zweck dieses Projekts ist es, **Leistung** und **Trainingsqualität** der beiden Bibliotheken im Kontext von Deep Learning zu bewerten. Dabei werden -so weit es geht- identische Modelle und Datensätze verwendet, um aussagekräftige und sinnvolle Vergleiche zu ermöglichen.

### Inhalt
- `/opennn/`: Implementierung und Experimente mit OpenNN
- `/mlpack/`: Implementierung und Experimente mit MLPack
- `/data/`: Verwendete Datensätze
- `/results/`: Metriken, Trainingszeiten, Plots

### Vergleichskriterien
| Kriterium           | OpenNN                             | MLPack                                |
|---------------------|-------------------------------------|----------------------------------------|
| API-Stil            | Objektorientiert, XML-Config       | Funktional, Template-basiert           |
| Dokumentation       | Gut, aber teilweise veraltet       | Umfangreich & aktiv gepflegt           |
| Training Performance| Schnell, Fokus auf CPU             | Sehr schnell, nutzt Armadillo          |
| Modellkomplexität   | Begrenzter Layer-Typ-Support       | Umfangreicher (CNNs, RNNs etc.)        |
| Deployment          | Reine C++-Abhängigkeit             | Leichtgewichtig, optional Bindings     |

### Voraussetzungen
- C++17 oder höher
- CMake >= 3.10
- Armadillo (für MLPack)
- Eigen (für OpenNN)

### Benchmark-Ergebnisse für Iris-Datensatz
#### Daten
- Wikipedia: https://de.wikipedia.org/wiki/Schwertlilien-Datensatz
- UCI ML Repository: https://archive.ics.uci.edu/dataset/53/iris
- Kaggle: https://www.kaggle.com/datasets/uciml/iris

#### Modell
- Feed-Forward NN (Multilayer Perceptron) mit 3 Schichten
  1. Input-Layer mit 4 Knoten
  2. Hidden-Layer mit 6 Knoten, ReLU
  3. Output-Layer mit 3 Knoten, Softmax

#### Löser
- ADAM
- Batch Size: 32
- Lernrate: 0.01
- Epochen: 500

#### Ergebnis
| Größe               | MLPack  | OpenNN  |
|---------------------|---------|---------|
| Training Time (ms)  | 273     | 1062    |
| Training Accuracy   | 99.17%  | 82.22%  |
| Training MSE        | 0.0210  | 0.0939  |
| Testing Accuracy    | 96.67%  | 76.67%  |
| Testing MSE         | 0.0393  | -       |

### Benchmark-Ergebnisse für Wine-Datensatz
#### Daten
- UCI ML Repository: https://archive.ics.uci.edu/dataset/109/wine
- Kaggle: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset

#### Modell
- Feed-Forward NN mit 5 Schichten (13-16-9-6-3)

#### Löser
- ADAM
- Batch Size: 32
- Lernrate: 0.01
- Epochen: 500

#### Ergebnis
| Größe               | MLPack      | OpenNN  |
|---------------------|-------------|---------|
| Training Time (ms)  | 595         | 1414    |
| Training Accuracy   | 100%        | 79.63%  |
| Training MSE        | 1.0946e-05  | 0.0915  |
| Testing Accuracy    | 97.22%      | 65.71%  |
| Testing MSE         | 0.0220      | -       |

### Fazit
- **OpenNN** eignet sich gut für kleinere, CPU-zentrierte Projekte.
- **MLPack** bietet mehr Flexibilität und bessere Performance für komplexere Modelle.

---

## Español

### OpenNN vs. MLPack – Comparación de bibliotecas de redes neuronales en C++
Este repositorio documenta una comparación práctica entre dos bibliotecas populares de aprendizaje automático en C++: [OpenNN](https://www.opennn.net/) y [MLPack](https://www.mlpack.org/), con enfoque en redes neuronales (Perceptrones Multicapa).

### Objetivo
El propósito de este proyecto es evaluar el **rendimiento** y la **calidad del entrenamiento** de ambas bibliotecas en el contexto del aprendizaje profundo. Se utilizan modelos y conjuntos de datos idénticos siempre que sea posible para garantizar comparaciones significativas.

### Contenido
- `/opennn/`: Implementación y experimentos con OpenNN  
- `/mlpack/`: Implementación y experimentos con MLPack  
- `/data/`: Conjuntos de datos utilizados  
- `/results/`: Métricas, tiempos de entrenamiento, gráficos  

### Criterios de Comparación
| Criterio            | OpenNN                               | MLPack                                 |
|---------------------|---------------------------------------|----------------------------------------|
| Estilo de API       | Orientado a objetos, configuración XML| Funcional, basado en plantillas        |
| Documentación       | Buena, algo desactualizada           | Amplia y activamente mantenida         |
| Rendimiento         | Rápido, centrado en CPU              | Muy rápido, usa Armadillo              |
| Complejidad del modelo | Soporte limitado de capas          | Más completo (CNNs, RNNs, etc.)        |
| Despliegue          | Dependencia sólo de C++              | Ligero, enlaces opcionales             |

### Requisitos Previos
- C++17 o superior  
- CMake >= 3.10  
- Armadillo (para MLPack)  
- Eigen (para OpenNN)  

### Resultados del Benchmark – Conjunto de datos Iris
#### Datos
- Wikipedia: https://es.wikipedia.org/wiki/Conjunto_de_datos_de_las_iris  
- UCI ML Repository: https://archive.ics.uci.edu/dataset/53/iris  
- Kaggle: https://www.kaggle.com/datasets/uciml/iris  

#### Modelo
- Red neuronal (Perceptrón Multicapa) de 3 capas:
  1. Capa de entrada con 4 nodos  
  2. Capa oculta con 6 nodos, ReLU  
  3. Capa de salida con 3 nodos (targets one-hot), Softmax  

#### Optimizador
- ADAM  
- Tamaño del lote: 32  
- Tasa de aprendizaje inicial: 0.01  
- Épocas: 500  

#### Resultado
| Métrica            | MLPack  | OpenNN  |
|--------------------|---------|---------|
| Tiempo Entrenamiento (ms) | 273     | 1062    |
| Precisión Entrenamiento   | 99.17%  | 82.22%  |
| MSE Entrenamiento         | 0.0210  | 0.0939  |
| Precisión Prueba          | 96.67%  | 76.67%  |
| MSE Prueba                | 0.0393  | -       |

### Resultados del Benchmark – Conjunto de datos Wine
#### Datos
- UCI ML Repository: https://archive.ics.uci.edu/dataset/109/wine  
- Kaggle: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset  

#### Modelo
- Red neuronal (Perceptrón Multicapa) de 5 capas (13-16-9-6-3)

#### Optimizador
- ADAM  
- Tamaño del lote: 32  
- Tasa de aprendizaje inicial: 0.01  
- Épocas: 500  

#### Resultado
| Métrica            | MLPack     | OpenNN  |
|--------------------|------------|---------|
| Tiempo Entrenamiento (ms) | 595        | 1414    |
| Precisión Entrenamiento   | 100%       | 79.63%  |
| MSE Entrenamiento         | 1.0946e-05 | 0.0915  |
| Precisión Prueba          | 97.22%     | 65.71%  |
| MSE Prueba                | 0.0220     | -       |

### Conclusión
- **OpenNN** es adecuado para proyectos más pequeños centrados en CPU.  
- **MLPack** ofrece mayor flexibilidad y mejor rendimiento en modelos complejos.

---

## OMG

### OpenNN vs. MLPack – Comparison of Neural Network Libraries in C++
This repository documents a practical comparison between two popular C++ machine learning libraries...

[Truncated for space – will continue in next steps]

### Objective
The goal of this project is to evaluate **performance** and **training quality** of both libraries in the context of deep learning. Identical models and datasets are used as far as possible to ensure meaningful comparisons.

### Content
- `/opennn/`: Implementation and experiments with OpenNN  
- `/mlpack/`: Implementation and experiments with MLPack  
- `/data/`: Used datasets  
- `/results/`: Metrics, training times, plots  

### Comparison Criteria
| Criterion            | OpenNN                              | MLPack                                 |
|----------------------|--------------------------------------|----------------------------------------|
| API Style            | Object-oriented, XML-based config   | Functional, template-based             |
| Documentation        | Good, partially outdated            | Extensive and actively maintained      |
| Training Performance | Fast, CPU-focused                   | Very fast, uses Armadillo              |
| Model Complexity     | Limited layer types supported       | More extensive (CNNs, RNNs, etc.)      |
| Deployment           | Pure C++ dependency                 | Lightweight, optional bindings         |

### Prerequisites
- C++17 or higher  
- CMake >= 3.10  
- Armadillo (for MLPack)  
- Eigen (for OpenNN)  

### Benchmark Results for Iris Dataset
#### Data
- Wikipedia: https://en.wikipedia.org/wiki/Iris_flower_data_set  
- UCI ML Repository: https://archive.ics.uci.edu/dataset/53/iris  
- Kaggle: https://www.kaggle.com/datasets/uciml/iris  

#### Model
- Feed-forward NN (Multilayer Perceptron) with 3 layers:
  1. Input layer with 4 nodes  
  2. Hidden layer with 6 nodes, ReLU  
  3. Output layer with 3 nodes (one-hot targets), Softmax  

#### Solver
- ADAM  
- Batch size: 32  
- Initial learning rate: 0.01  
- Epochs: 500  

#### Result
| Metric             | MLPack  | OpenNN  |
|--------------------|---------|---------|
| Training Time (ms) | 273     | 1062    |
| Training Accuracy  | 99.17%  | 82.22%  |
| Training MSE       | 0.0210  | 0.0939  |
| Testing Accuracy   | 96.67%  | 76.67%  |
| Testing MSE        | 0.0393  | -       |

### Benchmark Results for Wine Dataset
#### Data
- UCI ML Repository: https://archive.ics.uci.edu/dataset/109/wine  
- Kaggle: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset  

#### Model
- Feed-forward NN with 5 layers (13-16-9-6-3)

#### Solver
- ADAM  
- Batch size: 32  
- Initial learning rate: 0.01  
- Epochs: 500  

#### Result
| Metric             | MLPack     | OpenNN  |
|--------------------|------------|---------|
| Training Time (ms) | 595        | 1414    |
| Training Accuracy  | 100%       | 79.63%  |
| Training MSE       | 1.0946e-05 | 0.0915  |
| Testing Accuracy   | 97.22%     | 65.71%  |
| Testing MSE        | 0.0220     | -       |

### Conclusion
- **OpenNN** is well-suited for smaller, CPU-focused projects.  
- **MLPack** offers more flexibility and better performance for more complex models.
