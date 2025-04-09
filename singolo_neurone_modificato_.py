import numpy as np

class Neurone:
    def __init__(self, numero_ingressi):
        """
        Inizializza un neurone artificiale.

        Args:
            numero_ingressi (int): Il numero di ingressi del neurone.
        """
        # Inizializza i pesi in modo casuale tra -1 e 1
        self.pesi = np.random.uniform(-1, 1, numero_ingressi)
        # Inizializza la bias (soglia) a 0
        self.bias = 0

    def calcola_uscita(self, ingressi):
        """
        Calcola l'uscita del neurone per gli ingressi forniti.

        Args:
            ingressi (list o numpy.ndarray): Una lista o un array NumPy contenente i valori degli ingressi.

        Returns:
            int: 0 o 1, l'uscita del neurone dopo l'applicazione della funzione di attivazione.
        """
        # Calcola la somma ponderata degli ingressi e della bias
        somma_ponderata = np.dot(ingressi, self.pesi) + self.bias
        #somma_ponderata = sum(ingressi*self.pesi) + self.bias
        # Applica la funzione di attivazione (funzione gradino o Heaviside)
        uscita = 1 if somma_ponderata >= 0 else 0
        return uscita

    def aggiorna_pesi(self, ingressi, errore, tasso_apprendimento):
        """
        Aggiorna i pesi del neurone in base all'errore.

        Args:
            ingressi (list o numpy.ndarray): I valori degli ingressi utilizzati per la previsione.
            errore (int): L'errore tra l'uscita desiderata e l'uscita calcolata.
            tasso_apprendimento (float): La velocità con cui i pesi vengono aggiustati.
        """
        self.pesi += tasso_apprendimento * errore * np.array(ingressi)
        self.bias += tasso_apprendimento * errore

def collauda_neurone(neurone, dati_test):
    """
    Collauda il neurone con un set di dati di test.

    Args:
        neurone (Neurone): L'istanza del neurone da collaudare.
        dati_test (list di tuple): Una lista di tuple, dove ogni tupla contiene
                                    gli ingressi (lista o numpy.ndarray) e l'uscita desiderata (int).
    """
    print("\n--- Collaudo del Neurone ---")
    for ingressi, uscita_desiderata in dati_test:
        uscita_calcolata = neurone.calcola_uscita(ingressi)
        print(f"Ingressi: {ingressi}, Uscita Calcolata: {uscita_calcolata}, Uscita Desiderata: {uscita_desiderata}")

if __name__ == "__main__":
    # Definisci il numero di ingressi del neurone
    numero_ingressi = 2

    # Crea un'istanza del neurone
    neurone = Neurone(numero_ingressi)
    print("Neurone creato con pesi iniziali:", neurone.pesi, "e bias:", neurone.bias)

    # Definisci alcuni dati di test (esempio per la funzione logica AND)
    dati_test_and = [	#è una lista, ogni elemento è una tupla che contiene due elementi,                       
        ([0, 0], 0), 	#lista ingressi ed uscita desiderata
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 1)
    ]
    dati_test_and_2 = [	#è una lista, ogni elemento è una tupla che contiene due elementi,                       
        ([0.2, 0.1], 0), 	#lista ingressi ed uscita desiderata
        ([0.3, 1.1], 1),
        ([1.2, 0.3], 1),
        ([1.1, 0.9], 1)
    ]  
    

    print(" Collauda il neurone prima dell'apprendimento")
    collauda_neurone(neurone, dati_test_and)

    # --- Fase di Apprendimento (opzionale per la verifica di base) ---
    # Questa sezione è inclusa per mostrare come si potrebbe addestrare il neurone
    # ma per la semplice verifica dell'output 0 o 1 non è strettamente necessaria.
    print("\n--- Inizio Fase di Apprendimento (semplice esempio) ---")
    tasso_apprendimento = 0.1
    epoche = 20

    for epoca in range(epoche):
        print(f"\nEpoca {epoca + 1}")
        for ingressi, uscita_desiderata in dati_test_and:
            uscita_calcolata = neurone.calcola_uscita(ingressi)
            errore = uscita_desiderata - uscita_calcolata
            print(f"Ingressi: {ingressi}, Uscita Calcolata: {uscita_calcolata}, Desiderata: {uscita_desiderata}, Errore: {errore}")
            neurone.aggiorna_pesi(ingressi, errore, tasso_apprendimento)
        print("Pesi dopo l'epoca:", neurone.pesi, "Bias:", neurone.bias)

    print(" Collauda il neurone dopo l'apprendimento (se la fase di apprendimento è stata eseguita)")
    collauda_neurone(neurone, dati_test_and_2)
