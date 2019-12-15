---
title: 'Exposé'
disqus: hackmd
---

Motivation
---
Oftmals ist die Ausführung von Transportationsmodellen für Ozeane sehr aufwändig. Eine Möglichkeit die Ausführung stellt die TMM-Methode dar, die unter anderem von Metos3D benutzt wird. Die Erstellung der dafür notwendigen Transport-Matrizen ist allerdings sehr aufwändig und kann deswegen nicht leicht für neue Ozeanmodelle realisiert werden. Deshalb soll ein neuronales Netzwerk trainiert werden können, um die Transportation zu approximieren.

Funktionsweise von TMM
---
Es gibt Matrizen $A_1, ..., A_{12}$. Für den Monat $i$ stellt die Matriz $A_1$ die durchschnittliche Strömung des Monats dar. Je nach Tag im Monat fließen die Matrizen unterschiedlich stark ein.

$$\alpha \in (0, 1)$$
$$x_{k+1} = B \cdot x_k$$
$$B = \alpha A_i + (1-\alpha) A_{i+1}$$

Gewünschter Ablauf mit Verwendung des Netzwerks
---
Die Funktion $f$ stellt im Folgenden die Anwendung des Netzwerks zur Simulation des Transports dar. Die Funktion $g$ stellt die Anwendung eines Biomodells dar.

$$\tilde{x}_{k+1} = f(x_k)$$
$$x_{k+1} = g(\tilde{x}_{k+1})$$

Dabei soll Metos3D nicht mehr verwendet werden und mit dem kombinierten Modell (Netzwerk und Biomodell) auch der Spinup durchgeführt werden.

Alternativer Ansatz
---
Alternativ könnte ein Modell benutzt werden, dass anhand von Transportdaten die Transportmatrizen bestimmt. Diese Transportmatrizen könnten dann anschließend mit Metos3D verwendet werden, um den Transport zu approximieren. Dieser Ansatz soll allerdings nicht tiefergehend untersucht werden.

Technologie
---
Die Umsetzung des neuronalen Netzwerks soll mit Keras erfolgen. Als Backend für Keras wird Tensorflow gewählt. Die Daten werden mit netcdf4 eingelesen und falls nötig vorher aus petsc-Daten mit hdf5 konvertiert. Zur Darstellung der Daten wird matplotlib verwendet.

Zu erwartende Schwierigkeiten
---
Insbesondere die Tatsache, dass das Netzwerk sehr oft hintereinander angewendet werden soll und die Möglichkeit, dass die verwendeten Biomodelle sehr stark auf Abweichungen reagieren können, führt zu Schwierigkeiten. Es ist also sowohl wichtig, dass das Modell auf einem Schritt sehr geringe Abweichungen erzeugt, aber insbesondere auch, dass die erzeugten Fehler nicht explodieren. Es sollte auch aufgepasst werden, dass insbesondere die Massenerhaltung eingehalten wird.

Geplante Umsetzung
---
Die Umsetzung wird auf dem im Rahmen des Masterprojekts erarbeiteten Netzwerks basieren. Allerdings wird das Netzwerk angepasst werden, um die Performance zu verbessern, die Stabilität zu erhöhen und die Beschränkung auf ein reines Transportmodell zu berücksichtigen.

Trainingsdaten
---
Erste Trainingsdaten sollen mit dem Zero-Modell von Metos3D erzeugt werden und mit zufälligen Ausgangswerten arbeiten. Eventuell müssen Trainingsdaten erzeugt werden, bei denen ein gekoppeltes Biomodell verwendet wurde, um die Trainingsdaten auf realistische Bereiche zu begrenzen.

Geplante Meilensteine
---
- Erzeugung eines Netzwerks zur Berechnung der Zeitschritte eines Monats (mit geringen Abweichungen und ohne Biomodell)
- Erzeugung eines oder mehrerer Netzwerke zur Berechnung der Zeitschritte eines Jahres (mit geringen Abweichungen und ohne Biomodell)

Erfolg
---
- Zufriedenstellend: Geringe Unterschiede der Daten nach Laufzeit von einem Jahr für ein reines Transportmodell (basierend auf Zero-Modell von Metos3D)
- Ideal: Geringe Unterschiede der Daten nach kompletter Laufzeit (inklusive Spinup) für ein Modell mit Transportmodell basierend auf Metos3D und angeschlossenem Biomodell
- Weiterführend: Nutzbarkeit der Netzwerk-Topologie für Approximation mehrerer Transportmodelle (Training pro Modell ist in Ordnung)
