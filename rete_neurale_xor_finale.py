import numpy as np

class Neurone:
    def __init__(self, num_inputs):
        self.pesi = np.random.rand(num_inputs)
        self.bias = np.random.rand(1)[0]
        self.output = 0

    def sigmoide(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoide_derivata(self, x):
        return x * (1 - x)

    def calcola_output(self, inputs):
        self.input_corrente = inputs
        somma_ponderata = np.dot(inputs, self.pesi) + self.bias
        self.output = self.sigmoide(somma_ponderata)
        return self.output

    def aggiorna_pesi(self, delta_pesi, tasso_apprendimento):
        self.pesi += delta_pesi * tasso_apprendimento

    def aggiorna_bias(self, delta_bias, tasso_apprendimento):
        self.bias += delta_bias * tasso_apprendimento

class ReteNeurale:
    def __init__(self, num_input, num_nascosti, num_output):
        self.neuroni_nascosti = [Neurone(num_input) for _ in range(num_nascosti)]
        self.neurone_output = Neurone(num_nascosti)

    def feedforward(self, input_data):
        output_nascosti = [neurone.calcola_output(input_data) for neurone in self.neuroni_nascosti]
        output_finale = self.neurone_output.calcola_output(output_nascosti)
        return output_finale, output_nascosti

    def retropropagazione(self, input_data, output_desiderato, output_reale, output_nascosti):
        # Errore dello strato di output
        errore_output = output_desiderato - output_reale
        delta_output = errore_output * self.neurone_output.sigmoide_derivata(output_reale)

        # Errore dello strato nascosto
        delta_nascosti = np.zeros(len(self.neuroni_nascosti))
        for i, neurone_nascosto in enumerate(self.neuroni_nascosti):
            delta_nascosti[i] = delta_output * self.neurone_output.pesi[i] * neurone_nascosto.sigmoide_derivata(output_nascosti[i])

        # Calcolo delle modifiche ai pesi e ai bias per lo strato di output
        delta_pesi_output = np.array(output_nascosti) * delta_output
        delta_bias_output = delta_output

        # Calcolo delle modifiche ai pesi e ai bias per lo strato nascosto
        delta_pesi_nascosti = []
        for i, neurone_nascosto in enumerate(self.neuroni_nascosti):
            delta_pesi_nascosti.append(input_data * delta_nascosti[i])
        delta_bias_nascosti = delta_nascosti

        return delta_pesi_output, delta_bias_output, delta_pesi_nascosti, delta_bias_nascosti

    def fit(self, input_dati, output_desiderati, tasso_apprendimento=0.1, epoche=10000):
        num_samples = len(input_dati)

        for epoca in range(epoche): #epoche
            errore_totale = 0
            for i in range(num_samples):
                input_corrente = input_dati[i]
                output_desiderato = output_desiderati[i]

                # Feedforward
                output_reale, output_nascosti = self.feedforward(input_corrente)

                # Calcolo dell'errore
                errore = 0.5 * (output_desiderato - output_reale) ** 2
                errore_totale += errore

                # Retropropagazione
                delta_pesi_output, delta_bias_output, delta_pesi_nascosti, delta_bias_nascosti = self.retropropagazione(
                    input_corrente, output_desiderato, output_reale, output_nascosti
                )

                # Aggiornamento dei pesi e dei bias
                self.neurone_output.aggiorna_pesi(delta_pesi_output, tasso_apprendimento)
                self.neurone_output.aggiorna_bias(delta_bias_output, tasso_apprendimento)
                for j, neurone_nascosto in enumerate(self.neuroni_nascosti):
                    neurone_nascosto.aggiorna_pesi(delta_pesi_nascosti[j], tasso_apprendimento)
                    neurone_nascosto.aggiorna_bias(delta_bias_nascosti[j], tasso_apprendimento)

            if (epoca + 1) % 1000 == 0:
                print(f"Epoca {epoca + 1}/{epoche}, Errore: {errore_totale / num_samples:.4f}")

    def predici(self, input_data):
        output_nascosti = [neurone.calcola_output(input_data) for neurone in self.neuroni_nascosti]
        return self.neurone_output.calcola_output(output_nascosti)

if __name__ == '__main__':
    # Dati di addestramento per la funzione XOR
    input_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    output_xor = np.array([0, 1, 1, 0])

    # Creazione della rete neurale
    rete_xor = ReteNeurale(num_input=2, num_nascosti=2, num_output=1)

    # Addestramento della rete
    rete_xor.fit(input_xor, output_xor, tasso_apprendimento=0.5, epoche=15000)

    
    # Test della rete
    print("\nRisultati dopo l'addestramento:")
    for i in range(len(input_xor)):
        predizione = rete_xor.predici(input_xor[i])
        print(f"Input: {input_xor[i]}, Predizione: {predizione:.4f}, Desiderato: {output_xor[i]}")
    