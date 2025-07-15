

**Language / Sprache / Idioma:**  
[🇩🇪 Deutsch](#deutsch) | [🇬🇧 English](#english) |  [🇪🇸 Español](#espa%C3%B1ol)

---

#### _Deutsch_

## OpenNN vs. MLPack – Vergleich neuronaler Netzwerkbibliotheken in C++
Dieses Repository dokumentiert einen praktischen Vergleich zwischen zwei populären C++-Bibliotheken für maschinelles Lernen: [OpenNN](https://www.opennn.net/) und [MLPack](https://www.mlpack.org/), mit Fokus auf neuronale Netzwerke (Multilayer Perceptrons).

### Zielsetzung

Der Zweck dieses Projekts ist es, **Leistung**, **Benutzerfreundlichkeit**, **Flexibilität** und **Trainingsqualität** der beiden Bibliotheken im Kontext von Deep Learning zu bewerten. Dabei werden identische Modelle und Datensätze verwendet, um faire Vergleiche zu ermöglichen.

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

### Benchmark-Ergebnisse (Beispiel)

| Modell      | Datensatz | Bibliothek | Genauigkeit | Trainingszeit |
|-------------|-----------|------------|-------------|----------------|
| MLP (2 Lagen)| Iris      | OpenNN     | 96,7 %      | 0,8 s          |
| MLP (2 Lagen)| Iris      | MLPack     | 97,3 %      | 0,3 s          |
| MLP (3 Lagen)| MNIST     | OpenNN     | 91,2 %      | 34,5 s         |
| MLP (3 Lagen)| MNIST     | MLPack     | 92,5 %      | 21,7 s         |

### Fazit

- **OpenNN** eignet sich gut für kleinere, CPU-zentrierte Projekte.
- **MLPack** bietet mehr Flexibilität und bessere Performance für komplexere Modelle.

### Lizenz

<!--MIT-Lizenz-->

---

#### _English_

## OpenNN vs. MLPack – Comparison of Neural Network Libraries in C++

This repository documents a practical comparison between two popular C++ machine learning libraries: [OpenNN](https://www.opennn.net/) and [MLPack](https://www.mlpack.org/), with a focus on neural networks (Multilayer Perceptrons).

### Objective

The goal is to compare **performance**, **usability**, **flexibility**, and **training quality** of both libraries using identical models and datasets for fairness.

### Contents

- `/opennn/`: Implementation and experiments with OpenNN
- `/mlpack/`: Implementation and experiments with MLPack
- `/data/`: Datasets used
- `/results/`: Metrics, training times, plots

### Comparison Criteria

| Criterion           | OpenNN                             | MLPack                                |
|---------------------|-------------------------------------|----------------------------------------|
| API Style           | Object-oriented, XML config         | Functional, template-based             |
| Documentation       | Good, partially outdated            | Extensive & actively maintained        |
| Training Performance| Fast, CPU-focused                   | Very fast, uses Armadillo              |
| Model Complexity    | Limited layer types                 | Richer support (CNNs, RNNs etc.)       |
| Deployment          | Pure C++ dependency                 | Lightweight, optional bindings         |

### Requirements

- C++17 or newer
- CMake >= 3.10
- Armadillo (for MLPack)
- Eigen (for OpenNN)

### Benchmark Results (Example)

| Model       | Dataset  | Library | Accuracy | Training Time |
|-------------|----------|---------|----------|----------------|
| MLP (2-layers)| Iris    | OpenNN  | 96.7%    | 0.8s           |
| MLP (2-layers)| Iris    | MLPack  | 97.3%    | 0.3s           |
| MLP (3-layers)| MNIST   | OpenNN  | 91.2%    | 34.5s          |
| MLP (3-layers)| MNIST   | MLPack  | 92.5%    | 21.7s          |

### Conclusion

- **OpenNN** is suitable for small, CPU-oriented projects.
- **MLPack** offers more flexibility and better performance for complex models.

### License

<!--MIT License-->

---

#### _Español_

## OpenNN vs. MLPack – Comparación de bibliotecas de redes neuronales en C++

Este repositorio documenta una comparación práctica entre dos bibliotecas populares de aprendizaje automático en C++: [OpenNN](https://www.opennn.net/) y [MLPack](https://www.mlpack.org/), enfocándose en redes neuronales (perceptrones multicapa).

## Objetivo

El objetivo es comparar el **rendimiento**, la **usabilidad**, la **flexibilidad** y la **calidad del entrenamiento** de ambas bibliotecas usando modelos y conjuntos de datos idénticos para asegurar una comparación justa.

## Contenido

- `/opennn/`: Implementación y experimentos con OpenNN
- `/mlpack/`: Implementación y experimentos con MLPack
- `/data/`: Conjuntos de datos utilizados
- `/results/`: Métricas, tiempos de entrenamiento, gráficos

## Criterios de Comparación

| Criterio            | OpenNN                             | MLPack                                |
|---------------------|-------------------------------------|----------------------------------------|
| Estilo de API       | Orientado a objetos, config XML     | Funcional, basado en templates         |
| Documentación       | Buena, algo desactualizada          | Extensa y mantenida activamente        |
| Rendimiento         | Rápido, orientado a CPU             | Muy rápido, usa Armadillo              |
| Complejidad del modelo| Tipos de capas limitados           | Más variedad (CNNs, RNNs, etc.)        |
| Despliegue          | Solo depende de C++                 | Ligera, con enlaces opcionales         |

## Requisitos

- C++17 o superior
- CMake >= 3.10
- Armadillo (para MLPack)
- Eigen (para OpenNN)

## Resultados de Benchmark (Ejemplo)

| Modelo       | Dataset | Biblioteca | Precisión | Tiempo de Entrenamiento |
|--------------|---------|------------|-----------|--------------------------|
| MLP (2 capas)| Iris    | OpenNN     | 96.7%     | 0.8s                     |
| MLP (2 capas)| Iris    | MLPack     | 97.3%     | 0.3s                     |
| MLP (3 capas)| MNIST   | OpenNN     | 91.2%     | 34.5s                    |
| MLP (3 capas)| MNIST   | MLPack     | 92.5%     | 21.7s                    |

## Conclusión

- **OpenNN** es útil para proyectos pequeños orientados a CPU.
- **MLPack** proporciona mayor flexibilidad y rendimiento para modelos más complejos.

## Licencia

<!--Licencia MIT-->