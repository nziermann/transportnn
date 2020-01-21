# Modell

## Aufgabe
Es soll Metos3D zur Simulation des reinen Transports abgelöst werden. Die Simulation eines Biomodells ist **nicht** Teil der Aufgabe.
Die Funktion $f$ stellt im Folgenden die Anwendung des Netzwerks zur Simulation des Transports dar. Die Funktion $g$ stellt die Anwendung eines Biomodells dar. Dann sieht die gewünschte Funktionsweise folgendermaßen aus:

$$
\tilde{x}_{k+1} = f(x_k) \\
x_{k+1} = g(\tilde{x}_{k+1})
$$

Mit dem kombinierten Modell (Netzwerk und Biomodell) soll auch der Spinup durchgeführt werden können.

## Trainingsdaten
Wir erzeugen die Trainingsdaten folgendermaßen:
```graphviz
digraph Datenerzeugung {
    RandomDataGeneration[shape="box"];
    Normalisierung[shape="box"];
    "Metos3D-Spinup(1)"[shape="box"];
    "Metos3D-Spinup(2)"[shape="box"];
     RandomDataGeneration->random_data->Normalisierung->normalized_data->"Metos3D-Spinup(1)"->spinup_data->"Metos3D-Spinup(2)"->training_data
}
```

Die Normalisierung sorgt für eine realistische Gesamtmasse der einzelnen Stoffe. Der erste Metos3D-Spinup sorgt für eine realistische Verteilung der Stoffe, da eine zufällige Verteilung viele sehr scharfe Kanten bietet, die in echt nicht auftreten.

## Visualisierung
 Zur Visualisierung benutzen wir Panoply. Wir erzeugen entsprechende NetCDF-Dateien mit den folgenden Daten:
  - original
  - diff
  - model_prediction

  Durch die Visualisierung durch Panoply kann dabei sehr gut der Output des Netzwerks nachvollzogen werden. 


## Probierte Modelle
- RNN
- CNN (3-D)
- CNN (1-D)

## CNN (3-D)
Folgende Features wurden erarbeitet und benutzt:
- LandRemoval
- MassConversation

### Topologien
```graphviz
digraph autoencoder {
     subgraph overview {
       I [label="input"]
       "Land-Removal(input)"[shape="box"]
     
       M1 [label="1. minimilization layer", shape="box"]
       I->"Land-Removal(i)"->"intermediate_values(1)"->M1
     
       M2 [label="2. minimilization layer", shape="box"]
       M1->"intermediate_values(2)"->M2
    
       U1 [label = "1. Upsampling-Layer", shape="box"]
       M2->"intermediate_values(3)"->U1
    
       "Land-Removal(values)"[shape="box"]
       U2 [label = "2. Upsampling-Layer", shape="box"]
       U1->"intermediate_values(4)"->U2
       label="Overview";
       graph[style=box];
       U2->"intermediate_values(5)"->"Land-Removal(values)"->values
       
       Massenerhaltung[shape="box"]
       I->Massenerhaltung
       values->Massenerhaltung->output
       
     }

     subgraph cluster_min {
       NM [label="Input"]

       CM [label="convolutional layer", shape="box"]
       NM->CM

       PM [label = "Pooling-Layer", shape="box"]
       CM ->"intermediate_values(m)"->PM->"values(m)"
       label="Minimization layer";
       graph[style=box];
     }
     
     subgraph cluster_up {
       NU [label= "Input"]

       UU [label = "1. Upsampling-Layer", shape="box"]
       NU->UU

       CU [label="3. convolutional layer", shape="box"]
       UU->"intermediate_values(u)"->CU->"values(u)"

       label="Upsampling Layer";
       graph[style=box];
     }
   }
```

```graphviz
digraph simple {
    "Massenerhaltung"[shape="box"]
    "CNN-Layer(i)"[shape="box"]
    input -> "Massenerhaltung"
    input -> "CNN-Layer(i)"
    "CNN-Layer(i)"->values[label="i = n"]
    values->"Massenerhaltung"->output
    "CNN-Layer(i)"->intermediate_values[label="i < n"]
    intermediate_values->"CNN-Layer(i)"
}
```

## CNN (1-D)
Folgende Features wurden erarbeitet und benutzt:
- MassConversation

## Topologie
Die Struktur folgt der Struktur des CNN (3-D) bloß ohne Verwendung der Schicht LandRemoval.

