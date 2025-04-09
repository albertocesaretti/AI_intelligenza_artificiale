import numpy as np

def gradino(x): #funzione a gradino out 1 se x >0 altrimenti 0
    return 0 if x <=0 else 1

class Neurone:
    def __init__(self,bias, pesi, tasso_apprendimento,funzione_attivazione = gradino):
        self.bias = bias
        self.pesi = pesi #vettore con due valori
        self.tasso_apprendimento = tasso_apprendimento
        self.funzione_attivazione = funzione_attivazione 
        self.stato = 0
    
    def calcola_uscita(self, ingressi):
        
        somma_ponderata = np.dot(ingressi, self.pesi) + self.bias
        return self.funzione_attivazione(somma_ponderata)
    
    def aggiorna_pesi(self, ingressi, errore):
        
        self.pesi += self.tasso_apprendimento * errore * np.array(ingressi)
        self.bias += self.tasso_apprendimento * errore


def collauda_neurone(neurone, dati_test):
    print("\n--- Collaudo del Neurone ---")
    for ingressi, uscita_desiderata in dati_test:
        uscita_calcolata = neurone.calcola_uscita(ingressi)
        print(f"Ingressi: {ingressi}, Uscita Calcolata: {uscita_calcolata}, Uscita Desiderata: {uscita_desiderata}")

# Definisci alcuni dati di test (esempio per la funzione logica AND)
dati_test = [	#è una lista, ogni elemento è una tupla che contiene due elementi,                       
    ([0, 0], 0), 	#lista ingressi ed uscita desiderata
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1)
]

numero_ingressi = 2
bias = 0
pesi = np.random.uniform(-1, 1, numero_ingressi)
tasso_apprendimento = 0.1
epoche = 100

#creo il neurone
neurone = Neurone(bias, pesi, tasso_apprendimento,funzione_attivazione = gradino)
print("Neurone creato con pesi iniziali:", neurone.pesi, "e bias:", neurone.bias)

print(" Collauda il neurone prima dell'apprendimento")
collauda_neurone(neurone, dati_test)
print("************************************")
print("\n--- Inizio Fase di Apprendimento (semplice esempio) ---")
for epoca in range(epoche):
    print(f"\nEpoca {epoca + 1}")
    for ingressi, uscita_desiderata in dati_test: #primo giro ingressi = [0,0] uscita_desiderata = 0
        uscita_calcolata = neurone.calcola_uscita(ingressi)	#secondo giro ingressi = [0,1] uscita_desiderata = 0	
        errore = uscita_desiderata - uscita_calcolata		#quarto giro ingressi = [1,1] uscita_desiderata = 1
        print(f"Ingressi: {ingressi}, Uscita Calcolata: {uscita_calcolata}, Desiderata: {uscita_desiderata}, Errore: {errore}")
        neurone.aggiorna_pesi(ingressi, errore)
    print("Pesi dopo l'epoca:", neurone.pesi, "Bias:", neurone.bias)
print("************************************")
dati_test_modificati = [	#è una lista, ogni elemento è una tupla che contiene due elementi,                       
    ([0.1, 0.2], 0), 	#lista ingressi ed uscita desiderata
    ([0.2, 1.2], 0),
    ([1.1, 0.1], 0),
    ([1.3, 1.4], 1)
]
print(" Collauda il neurone dopo l'apprendimento con ingressi modificati e reali")
collauda_neurone(neurone, dati_test_modificati)