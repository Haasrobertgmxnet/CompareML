

**Language / Sprache / Idioma:**  
[🇩🇪 Deutsch](#deutsch) | [🇬🇧 English](#english) |  [🇪🇸 Español](#espa%C3%B1ol)

---

#### _Deutsch_

## OpenNN vs. MLPack – Vergleich neuronaler Netzwerkbibliotheken in C++
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
   * Wikipedia: https://de.wikipedia.org/wiki/Schwertlilien-Datensatz
   * UCI ML Repository: https://archive.ics.uci.edu/dataset/53/iris
   * Kaggle: https://www.kaggle.com/datasets/uciml/iris
   * Standard, geshuffled
	
#### Modell
   * Feed-Forward NN (Multilayer Perceptron) mit 3 Schichten
      1. Input-Layer mit 4 Knoten
	  2. Hidden-Layer mit 6 Knoten, Rectified Linear Unit (ReLU)
	  3. Output-Layer mit 3 Knoten (One-Hot-kodierte Targets), Softmax
	  
#### Löser
   * Adaptive Moment Estimation (ADAM)
   * Batch Size: 32
   * Initiale Lernrate: 0.01
   * Anzahl Epchen: 500
	  
#### Ergebnis

| Größe  | MLPack  | OpenNN  |
| Training Time (ms)  | 273  | 1062  |
| Training Accuracy  | 99.17%  | 82.22%  |
| Training MSE  | 0.0210  | 0.0939  |
| Testing Accuracy  | 96.67%  | 76.67%  |
| Testing MSE  | 0.0393  | -  |

### Benchmark-Ergebnisse für Wine-Datensatz

#### Daten
   * UCI ML Repository: https://archive.ics.uci.edu/dataset/109/wine
   * Kaggle: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
   * Standard, geshuffled
	
#### Modell
   * Feed-Forward NN (Multilayer Perceptron) mit 3 Schichten
      1. Input-Layer mit 13 Knoten
	  2. Hidden-Layer mit 16 Knoten, Rectified Linear Unit (ReLU)
	  3. Hidden-Layer mit 9 Knoten, Rectified Linear Unit (ReLU)
	  4. Hidden-Layer mit 6 Knoten, Rectified Linear Unit (ReLU)
	  5. Output-Layer mit 3 Knoten (One-Hot-kodierte Targets), Softmax
	  
#### Löser
   * Adaptive Moment Estimation (ADAM)
   * Batch Size: 32
   * Initiale Lernrate: 0.01
   * Anzahl Epchen: 500
	  
#### Ergebnis

| Größe  | MLPack  | OpenNN  |
| Training Time (ms)  | 595  | 1414  |
| Training Accuracy  | 100%  | 79.63%  |
| Training MSE  | 1.0946e-05  | 0.0915  |
| Testing Accuracy  | 97.22%  | 65.71%  |
| Testing MSE  | 0.0220  | -  |


### Fazit

- **OpenNN** eignet sich gut für kleinere, CPU-zentrierte Projekte.
- **MLPack** bietet mehr Flexibilität und bessere Performance für komplexere Modelle.

### Lizenz

<!--MIT-Lizenz-->

---

