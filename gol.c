/*
 ============================================================================

 Name:		mpi_gol.c
 Autoren:	Denis Maus & Yevgeniy Ruts
 Version:	2012-02-09

 Die Einbindung der MPI und SDL Header-Dateien ist je nach System unterschiedlich,
 s. dazu Kommentare weiter unten.

 MPI Implementation von "Game Of Life" => Conway's Spiel des Lebens
 - dynamische Spielrundenanzahl (Anzahl wird beim Programmstart abgefragt)
 - dynamisches Spielfeld (Dimension wird beim Programmstart abgefragt)
 - Spielfeld wird in Teilspielfelder aufgeteilt (ein Teil pro Prozess)
 - Grenzen (= Überlappung der Teilspielfelder) werden zwischen den Prozessen ausgetauscht
 - Berechnung und Ausgabe der Rechenzeit beim Programmende

 starten mit (N = Anzahl der Prozesse): mpirun -np N mpi_gol

 ============================================================================
 */

// MPI Bibliotheken
#include <mpi.h> // !!! je nach System, bei LINUX -> <mpi/mpi.h>
// Simple DirectMedia Layer Bibliothek
#include <SDL.h> // !!! je nach System, bei LINUX -> <SDL/SDL.h>
// Standart-Bibliotheken
#include <stdio.h>
#include <stdlib.h>

// Bibliothek für time()
#include <time.h>

// Bibliothek für u. A. memcpy()
#include <string.h>

// Bibliotheken für sleep() und usleep(); je nach Betriebssystem
#if WIN32
#include <sleep.h> // WINDOWS
#else
#include <unistd.h> // POSIX
#endif

// globale Difinitionen
#define ROOT 0 // Rootprozess-ID; ROOT hat das Vollspielfeld
// globale SDL Variablen
SDL_Surface *screen, *feld = NULL;
Uint32 cellColorLife, bgColor;
SDL_Rect cell;

/**
 * SDL Initialisieren
 */
void SDL_init(int _dim, int _cell_h);

/**
 * Fenster schließen und SDL beenden
 */
void GOL_gfx_gameover(void);

/**
 * Funktion zum allozieren von zusammenhängedem Speicher für ein 2D-Integer-Array
 * MPI-Funktionen arbeiten nur mit zusammenhängendem Speicher
 */
int **malloc2D(int _zeilen, int _spalten);

/**
 * Funktionen führen "Game Of Life" aus
 * "np = 1" -> startet serielle Version -> programmiert ohne MPI Funktionalität
 * "np > 1" -> startet parallele Version -> mit MPI Funktionen
 */
double GOL_parallel(int spielfeld_dim, int _runden,  MPI_Comm comm, int _speed, int _src_h, int _cell_h);
double GOL_seriell(int spielfeld_dim, int _runden, int _speed, int _scr_h, int _cell_h);

/**
 * grafische Ausgabe des Spielfeldes mit SDL
 */
void GOL_gfx(int **spielfeld, int _dim);

/**
 * MAIN
 *
 * 1) MPI initialisieren
 * 2) Abfrage der Spieldaten für Spielfeldgröße und Anzahl der Spielrunden
 * 3) Spiel starten
 * 4) gesamte Rechenzeit ausgeben
 * 5) MPI & Programm beenden
 *
 */
int main(int argc, char *argv[]) {
	int scr_h; // Bildschirmhöhe
	int cell_h; // Höhe der Zelle
	int my_rank; // eigene Proc-ID
	int dim; // Spielfelddimesnion dim x dim
	int runden; // Spielrundenanzahl
	int nprocs; // Prozessanzahl
	double zeit; // Ausführungszeit des Rechen-Algotithmus
	int speed; // Spielgeschwindigkeit, entspricht Intervall für das Leeren der Konsole
	char processorName[MPI_MAX_PROCESSOR_NAME]; // Processorname
	int processorNameMax, majorVersion, minorVersion; // Maximale Zeichenanzahl im Processornamen, MPI major- und minor-Version

	// MPI Initialisieren und Systeminfos abfragen
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Get_processor_name(processorName, &processorNameMax);
	MPI_Get_version(&majorVersion, &minorVersion);

	/**
	 * NUR ROOT-Prozess:
	 * - Abfrage von Spielrunden und Spielfeldgröße
	 */
	if (my_rank == ROOT) {
		// Anzeige serielle oder parallele Version
		printf("\n\033[7m----------------------------------------\033[m\n");
		if (nprocs == 1){
			printf("-[\033[31m serielle Version\033[m ]-[\033[1m 1 Prozess \033[m]-\n\n");
		}
		else {
			printf("%s\nMPI Version %d.%d bereit\n", processorName, majorVersion, minorVersion);
			printf("-[\033[31m parallele Version\033[m ]-[\033[1m %d Prozesse \033[m]-\n\n", nprocs);
		}

		// Spielparameter abfragen
		printf("Ihre Bildschirmhöhe eingeben \033[36m(z.B. 800)\033[m:\n");
		scanf("%d", &scr_h);
		printf("Die Größe einer Zelle eingeben \033[36m(z.B. 5)\033[m:\n");
		scanf("%d", &cell_h);
		printf("Die Dimension des Spielfelds eingeben \033[36m(z.B. 1000)\033[m:\n[\033[1m\033[31mDemomodus max. \033[5m%d\033[25m\033[m]\n", scr_h / cell_h);
		scanf("%d", &dim);
		printf("Die Anzahl der Spielrunden eingeben \033[36m(z.B. 100):\033[m \n");
		scanf("%d", &runden);
		printf("\n");

		/**
		 * Aktiviert ggf. den Demomodus
		 */
		if (dim <= scr_h / cell_h) {
			printf("\033[35m-[ Demomodus ]-\033[m\n");
			printf("Die Spielgeschwindigkeit in Millisekunden eingeben \033[36m(z.B. 500):\033[m \n");
			scanf("%d", &speed); // Spielgeschwindigkeit
			printf("\n");
			speed *= 1000; // für utime(): Millisekunden in Microsekunden umrechnen
			SDL_init(dim, cell_h); // Die grafische Schnittstelle initialisieren
		}
	}

	if (nprocs > 1) {
		/**
		 * Broadcast der relevanten Daten von Proc ROOT an alle anderen:
		 * runden = Spielrunden
		 * dim = Spielfelddimension (dim x dim)
		 */
		MPI_Bcast(&dim, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(&runden, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

		/**
		 * Game Of Life starten
		 * zeit = gesamte Rechnezeit
		 */
		printf("Proc %d: rechne ...\n", my_rank);
		zeit = GOL_parallel(dim, runden, MPI_COMM_WORLD, speed, scr_h, cell_h);
	} else if (nprocs == 1) {
		printf("Proc %d: rechne ...\n", my_rank);
		zeit = GOL_seriell(dim, runden, speed, scr_h, cell_h);
	}

	// Ausgabe der gesamten Rechenzeit
	if (my_rank == ROOT) {
		printf("\n\"Game Of Life\" beendet\nLaufzeit: \033[31m%lf Sekunden\033[m\n\033[7m----------------------------------------\033[m\n\n", zeit);
		// ggf. SDL beenden
		if (screen != NULL) {
			printf("Demomodus im Demofenster über Q- oder Escape-Taste bzw. Fenster schließen beenden...\n");
			GOL_gfx_gameover();
			printf("Demomodus erfolgreich beendet\n");
		}
	}

	// MPI beenden
	MPI_Finalize();

	// Programmende
	return EXIT_SUCCESS;
}

/**
 * Alloziert Speicherplatz für ein 2D-Integer-Array
 * _zeilen = Zeilen in Matrix
 * _spalten = Spalten in Matrix
 */
int **malloc2D(int _zeilen, int _spalten) {
	int i; // für Schleife
	int *daten = (int *) malloc(_zeilen * _spalten * sizeof(int)); // Pointer auf Speicherstelle mit (N x N)-Integer
	int **array = (int **) malloc(_zeilen * sizeof(int*)); // Pointer auf N-mal Integer Pointer
	for (i = 0; i < _zeilen; i++)
		array[i] = &(daten[_spalten * i]); // Jedem der vorher difiniertem Pointer einen Speicherberich festlegen
	return array; // Pinter die Speicherstelle mit NxN-Integer zurückgeben
}

/**
 * Funktion: parallele MPI-Version von "Game Of Life"
 * - zwei oder mehr Prozesse sind aktiv
 *
 * 1) Variablen Difinitionen
 * 2) MPI Daten abfragen
 * 3) Die Nachbarn OBEN und UNTEN bestimmen
 * 4) eigene Spielfeldgröße bestimmen (pro Prozess eigenes Teilspielfeld)
 * 5) Speicher für Spielfelder allozieren
 * 6) Spielfelder initialisieren (Anfangsbedingungen)
 * 7) Spiel runden-Mal spielen
 * 8) Rechenzeit aller Procs zusammentragen und zurückgeben
 *
 * Parameter:
 *  spielfeld_dim: Teilspielfeld X-Dimension
 *  _runden: Anzahl derSpielrunden
 *  _speed: Verzögerung zwischen Spielrunden
 *  _scr_h: Bildschirmhöhe
 *  _cell_h: Höhe der Zelle
 *  comm: MPI Kommunikator
 */
double GOL_parallel(int spielfeld_dim, int _runden, MPI_Comm comm, int _speed, int _scr_h, int _cell_h) {
	// Variablen
	int my_rank; // eigene Prozess-ID
	int nprocs; // Prozessanzahl
	int sende_oben, sende_unten, hole_oben, hole_unten; // Nachbarn OBEN und UNTEN
	int i, j, k; // für diverse Schleifen
	int my_size; // eigene Y_Teilspielfelddimension
	int my_elements; // Anzahl der Elemente in eigenem Teilspielfeld
	int my_offset; // Offset in dem Spielfeld
	int *all_size; // Size-Array mit my_size der einzelnen Procs
	int *all_elements; // Elements-Array mit my_elements der einzelnen Procs
	int *all_offset; // Offset-Array mit my_offset der einzelnen Procs
	int nachbarn; // Summe der Nachbarn
	int **my_spielfeld, **my_zwischen, **my_swap, **voll = NULL; // für Spielfelder
	double proc_zeit, max_zeit; // für Berechnung der Ausführungszeit -> Ausführungszeit jedes Prozesses und die längste Ausführungszeit

	// Prozessanzahl und eigene Prozess-ID bestimmen
	// (könnte auch über Parameter übergeben werden, aber Übersichtlichkeitshalber hier nochmal difiniert)
	MPI_Comm_size(comm, &nprocs);
	MPI_Comm_rank(comm, &my_rank);

	// Bestimmung der Nachbar-Teilspielfelder OBEN und UNTEN
	if (my_rank == ROOT) {
		sende_oben = hole_oben = MPI_PROC_NULL; // OBEN ist kein Teilspielfeld vorhanden
		sende_unten = hole_unten = my_rank + 1; // sende nach UNTEN und hole von UNTEN
	} else if (my_rank == nprocs - 1) {
		sende_unten = hole_unten = MPI_PROC_NULL; // UNTEN ist kein Teilspielfeld vorhanden
		sende_oben = hole_oben = nprocs - 2; // sende nach OBEN und hole von OBEN
	} else {
		sende_oben = hole_oben = my_rank - 1; // sende nach OBEN und hole von OBEN
		sende_unten = hole_unten = my_rank + 1; // sende nach UNTEN und hole von UNTEN
	}

	// eigene Y-Teilspielfeddimension bestimmen
	my_size = spielfeld_dim / nprocs + ((my_rank < (spielfeld_dim % nprocs)) ? 1 : 0);
	// eigene Anzahl der Elemente ohne obere und untere Zeile (Überlappung bzw. NULL-Zeile) bestimmen (jeder Proc für sich)
	my_elements = my_size * (spielfeld_dim + 2);

	// Offset bestimmen (jeder Proc für sich)
	my_offset = my_rank * (spielfeld_dim / nprocs);
	my_offset += (my_rank > (spielfeld_dim % nprocs)) ? (spielfeld_dim % nprocs) : my_rank;

	// NUR ROOT: alloziere dynamischen Speicehr für all_offset etc.
	if (my_rank == ROOT) {
		all_size = (int*) malloc(nprocs * sizeof(int));
		all_elements = (int*) malloc(nprocs * sizeof(int));
		all_offset = (int*) malloc(nprocs * sizeof(int));
	}

	// Zusammentragen im ROOT-Proc der berechneten Werte für my_size, my_elements und my_offset
	MPI_Gather(&my_size, 1, MPI_INT, all_size, 1, MPI_INT, ROOT, comm);
	MPI_Gather(&my_elements, 1, MPI_INT, all_elements, 1, MPI_INT, ROOT, comm);
	MPI_Gather(&my_offset, 1, MPI_INT, all_offset, 1, MPI_INT, ROOT, comm);

	/** FEHLERSUCHE
	 if (my_rank == ROOT) {
	 for (i = 0; i < nprocs; i++)
	 printf("all_size[%d] = %d // all_elements[%d] = %d // all_offset[%d] = %d\n", i, all_size[i], i, all_elements[i], i, all_offset[i]);
	 }
	 */

	// alloziere dynamischen Speicher für Teilspielfelder (jeder Proc für sich)
	my_spielfeld = malloc2D(my_size + 2, spielfeld_dim + 2);
	my_zwischen = malloc2D(my_size + 2, spielfeld_dim + 2);

	// NUR ROOT: alloziere dynamischen Speicher für Vollspielfeld
	if (my_rank == ROOT) {
		voll = malloc2D(spielfeld_dim, spielfeld_dim + 2);
	}

	/** FEHLERSUCHE
	 // jeder Proc gibt seinen allozierten Speicher aus
	 printf("Proc %d / my_spielfeld @ %p (%d Byte) / zwischen @ %p (%d Byte) / voll @ %p (%d Byte)\n", my_rank, my_spielfeld,
	 (int) ((&my_spielfeld[my_size + 1][matrix_size + 1] - &my_spielfeld[0][0] + 1) * sizeof(int)), my_zwischen,
	 (int) ((&my_zwischen[my_size + 1][matrix_size + 1] - &my_zwischen[0][0] + 1) * sizeof(int)), voll,
	 (voll != NULL ? (int) ((&voll[matrix_size + 1][matrix_size + 1] - &voll[0][0] + 1) * sizeof(int)) : 0));
	 */

	// Initialisierung des Aussenrands mit Nullen, entspricht kein Leben ausserhalb des Spielfelds
	for (i = 0; i < my_size + 2; i++)
		my_spielfeld[i][0] = my_spielfeld[i][spielfeld_dim + 1] = my_zwischen[i][0] = my_zwischen[i][spielfeld_dim + 1] = 0;
	for (j = 0; j < spielfeld_dim + 2; j++)
		my_spielfeld[0][j] = my_spielfeld[my_size + 1][j] = my_zwischen[0][j] = my_zwischen[my_size + 1][j] = 0;

	// für "echte" Zufallszhalen
	srand(my_rank + time(NULL));

	// Initialisierung des Spielfelds mit Werten, entspricht Anfangsbedingungen
	for (i = 1; i <= my_size; i++) {
		for (j = 1; j <= spielfeld_dim; j++)
			my_spielfeld[i][j] = rand() % 2;
	}

	MPI_Request request[4]; // für Isend/Irecv; Dimension = Anzahl der Isend/Irecv Operationen
	MPI_Status status[4]; // für Waitall; Dimension = Anzahl der Isend/Irecv Operationen

	// Startzeit speichern (jeder Proc für sich)
	proc_zeit = MPI_Wtime();

	// Spiel "_runden-Mal" spielen
	for (k = 0; k < _runden; k++) {
		// Sende und empfange Überlappung OBEN
		MPI_Isend(&my_spielfeld[1][0], spielfeld_dim + 2, MPI_INT, sende_oben, 0, MPI_COMM_WORLD, request);
		MPI_Irecv(&my_spielfeld[0][0], spielfeld_dim + 2, MPI_INT, hole_oben, 0, MPI_COMM_WORLD, request + 1);

		// Sende und empfange Überlappung UNTEN
		MPI_Isend(&my_spielfeld[my_size][0], spielfeld_dim + 2, MPI_INT, sende_unten, 0, MPI_COMM_WORLD, request + 2);
		MPI_Irecv(&my_spielfeld[my_size + 1][0], spielfeld_dim + 2, MPI_INT, hole_unten, 0, MPI_COMM_WORLD, request + 3);

		// Warte auf Abschluss von Isend/Irecv
		MPI_Waitall(4, request, status);

		/** FEHLERSUCHE
		 // Teilspielfelder mit der empfangenen Überlappungen ausgeben
		 printf("Anfangsfeld von %d:\n", my_rank);
		 for (i = 0; i < my_size + 2; i++) {
		 for (j = 0; j < matrix_size + 2; j++) {
		 printf("%d ", my_spielfeld[i][j]);
		 }
		 printf("\n");
		 }
		 */

		// Nachbarnanzahl in 8-ter Nachbarnschaft bestimen (jeder Proc für sich)
		for (i = 1; i <= my_size; i++) {
			for (j = 1; j < spielfeld_dim + 1; j++) {
				// Summe über alle Nachbarn bilden
				nachbarn = my_spielfeld[i - 1][j - 1] + my_spielfeld[i - 1][j] + my_spielfeld[i - 1][j + 1] + my_spielfeld[i][j - 1] + my_spielfeld[i][j + 1]
						+ my_spielfeld[i + 1][j - 1] + my_spielfeld[i + 1][j] + my_spielfeld[i + 1][j + 1];

				// leben, sterben, neues Leben ?
				if (nachbarn < 2 || nachbarn > 3) { // 0, 1, 4, 5, ...
					my_zwischen[i][j] = 0; // sterben
				} else if (nachbarn == 3) { // 3
					my_zwischen[i][j] = 1; // weiter leben bzw. neues Leben
				} else { // 2
					my_zwischen[i][j] = my_spielfeld[i][j]; // alten Wert übernehmen
				}
			}
		}

		// Umadressierung der Spielfelder
		my_swap = my_spielfeld; // Anfangsspielfeld -> Adresse in my_swap speichern
		my_spielfeld = my_zwischen; // Zwischenergebnis -> wird Endergebnis der aktuellen Spielrunde
		my_zwischen = my_swap; // Zwischenergebnis -> wird Anfangsspielfeld für die nächste Runde

		// Sende und empfange VOLL
		MPI_Status statusRecv;

		/**
		 * Proc ROOT empfängt die Teilspielfelder
		 * die Seitlichen NULL-Zeilen werden mitgesendet bzw. empfangen
		 * Überlappungen sowie obere und untere NULL-Zeilen werden nicht gesendet/empfangen
		 */
		if (my_rank == ROOT) {
			for (i = 1; i < nprocs; i++) {
				MPI_Recv(&voll[all_offset[i]][0], all_elements[i], MPI_INT, i, 99, MPI_COMM_WORLD, &statusRecv);
			}
		}

		// Teilspielfeld vom ROOT "schnell mal" ins das Vollspielfeld kopieren
		if (my_rank == ROOT) memcpy(&voll[0][0], &my_spielfeld[1][0], my_elements*sizeof(int));
		// Teilspielfelder anderer Procs an das Vollspielfeld senden
		else MPI_Send(&my_spielfeld[1][0], my_elements, MPI_INT, ROOT, 99, MPI_COMM_WORLD);

		// Ausgabe des Vollspielfelds in der Konsole
		if (my_rank == ROOT && screen != NULL) {
			GOL_gfx(voll, spielfeld_dim);
			usleep(_speed);
		}
	}

	// Rechnezeit berechnen und in proc_zeit speichern (jeder Proc für sich)
	proc_zeit = MPI_Wtime() - proc_zeit;

	// Die maximale Rechenzeite bestimmen und beim Proc 0 in max_zeit speichern
	MPI_Reduce(&proc_zeit, &max_zeit, 1, MPI_DOUBLE, MPI_MAX, ROOT, comm);

	// längste Rechenzeit zurückgeben
	return (max_zeit);
}

/**
 * Funktion: serielle Version von "Game Of Life"
 * - nur ein Prozess ist aktiv
 * - die MPI-Funktion MPI_Wtime() wird lediglich für die Messung der Ausführungszeit benutzt
 *
 * 1) Variablen Difinitionen
 * 2) Speicher für Spielfeld und Zwischenergebnis allozieren
 * 3) Spielfeld initialisieren (Anfangsbedingungen)
 * 4) Spiel runden-Mal spielen
 * 5) Rechenzeit zurückgeben
 *
 * Parameter:
 *  spielfeld_dim: Teilspielfeld X-Dimension
 *  _runden: Anzahl der Spielrunden
 *  _speed: Verzögerung zwischen Spielrunden
 *  _scr_h: Bildschirmhöhe
 *  _cell_h: Höhe der Zelle
 */
double GOL_seriell(int spielfeld_dim, int _runden, int _speed, int _scr_h, int _cell_h) {
	// Variablen
	int i, j, k; // für diverse Schleifen
	int nachbarn; // Summe der Nachbarn
	int **voll_zwischen, **voll_swap, **voll_spielfeld = NULL; // für Spielfelder
	double proc_zeit; // für Rechenzeitberechnung

	// alloziere dynamischen Speicher für Vollspielfeld und Zwischenergebnis
	voll_spielfeld = malloc2D(spielfeld_dim + 2, spielfeld_dim + 2);
	voll_zwischen = malloc2D(spielfeld_dim + 2, spielfeld_dim + 2);

	// Initialisierung des Aussenrands mit Nullen, entspricht kein Leben ausserhalb des Spielfelds
	for (i = 0; i < spielfeld_dim + 2; i++)
		voll_spielfeld[i][0] = voll_spielfeld[i][spielfeld_dim + 1] = voll_zwischen[i][0] = voll_zwischen[i][spielfeld_dim + 1] = 0;
	for (j = 0; j < spielfeld_dim + 2; j++)
		voll_spielfeld[0][j] = voll_spielfeld[spielfeld_dim + 1][j] = voll_zwischen[0][j] = voll_zwischen[spielfeld_dim + 1][j] = 0;

	// für "echte" Zufallszhalen
	srand(time(NULL));

	// Initialisierung des Spielfelds mit Werten, entspricht Anfangsbedingungen
	for (i = 1; i <= spielfeld_dim; i++) {
		for (j = 1; j <= spielfeld_dim; j++)
			voll_spielfeld[i][j] = rand() % 2;
	}

	// Startzeit speichern
	proc_zeit = MPI_Wtime();

	// Spiel "_runden-Mal" spielen
	for (k = 0; k < _runden; k++) {
		// Nachbarnanzahl in 8-ter Nachbarnschaft bestimen
		for (i = 1; i < spielfeld_dim + 1; i++) {
			for (j = 1; j < spielfeld_dim + 1; j++) {
				// Summe über alle Nachbarn bilden
				nachbarn = voll_spielfeld[i - 1][j - 1] + voll_spielfeld[i - 1][j] + voll_spielfeld[i - 1][j + 1] + voll_spielfeld[i][j - 1] + voll_spielfeld[i][j + 1]
						+ voll_spielfeld[i + 1][j - 1] + voll_spielfeld[i + 1][j] + voll_spielfeld[i + 1][j + 1];

				// leben, sterben, neues Leben ?
				if (nachbarn < 2 || nachbarn > 3) { // 0, 1, 4, 5, ...
					voll_zwischen[i][j] = 0; // sterben
				} else if (nachbarn == 3) { // 3
					voll_zwischen[i][j] = 1; // weiter leben bzw. neues Leben
				} else { // 2
					voll_zwischen[i][j] = voll_spielfeld[i][j]; // alten Wert übernehmen
				}
			}
		}

		// Umadressierung der Spielfelder
		voll_swap = voll_spielfeld; // Anfangsspielfeld -> Adresse in voll_swap speichern
		voll_spielfeld = voll_zwischen; // Zwischenergebnis -> wird Endergebnis der aktuellen Spielrunde
		voll_zwischen = voll_swap; // Zwischenergebnis -> wird Anfangsspielfeld für die nächste Runde

		// Grafische Ausgabe
		if (screen != NULL) {
			GOL_gfx(voll_spielfeld, spielfeld_dim);
			usleep(_speed);
		}
	}

	// Rechnezeit berechnen und in max_zeit speichern
	proc_zeit = MPI_Wtime() - proc_zeit;

	// gesamte Rechenzeit zurückgeben
	return (proc_zeit);
}

/**
 * Initialisierung SDL, Bildschirmfenster und Module
 */
void SDL_init(int _dim, int _cell_h) {
	// Video-Funktion initialisieren
	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		printf("Error: %s\n", SDL_GetError());
		exit(1);
	}
	atexit(SDL_Quit);

	// Bildschirmsurface initialisieren
	screen = SDL_SetVideoMode(_dim * _cell_h, _dim * _cell_h, 32, SDL_HWSURFACE | SDL_DOUBLEBUF);
	if (screen == NULL) {
		printf("Error: %s\n", SDL_GetError());
		exit(2);
	}
	SDL_WM_SetCaption("Game Of Life [von Denis Maus und Yevgeniy Ruts]", NULL); // Fenstertitel setzen

	cell.w = cell.h = _cell_h; // Größe einer Zelle difinieren (quadratisch)
	bgColor = SDL_MapRGB(screen->format, 0, 0, 0); // Farbe für Zelle setzen = schwarz
	cellColorLife = SDL_MapRGB(screen->format, 255, 0, 0); // Farbe für Zelle setzen = rot
}

/**
 * Grafische Ausgabe des Spielfeldes mit SDL
 */
void GOL_gfx(int **spielfeld, int _dim) {
	// Spielfeld mit Anfangsbedingungen initialisieren
	int i, j; // für Schleife

	// Hintergrundfarbe füllen
	SDL_FillRect(screen, NULL, bgColor);

	// Spielfeld zeichnen (lebende Zellen aufzeigen)
	for (i = 1; i < _dim; i++) {
		cell.y = (i - 1) * cell.h;
		for (j = 1; j < _dim; j++) {
			cell.x = (j - 1) * cell.w;
			if (spielfeld[i][j] == 1) {
				SDL_FillRect(screen, &cell, cellColorLife);
			}
		}
	}

	// Spielfeld zum Bildschirm "schicken"
	SDL_Flip(screen);
}

void GOL_gfx_gameover(void){
	SDL_Event event;
	int gameover=0;

	while (gameover == 0) {
			// Spiel gestartet
			while (SDL_PollEvent(&event)) {
				switch (event.type) {
					case SDL_MOUSEBUTTONDOWN: // Maustaste wurde gedrückt
						break;
					case SDL_KEYDOWN: // Tastaturtaste wurde gedrückt
						if (event.key.keysym.sym == SDLK_ESCAPE || event.key.keysym.sym == SDLK_q) {
							gameover = 1;
						}
						break;
					case SDL_QUIT:
						gameover = 1;
						break;
					default:
						break;
				}
			}
		}
	SDL_FreeSurface(screen);
	SDL_Quit();
}
