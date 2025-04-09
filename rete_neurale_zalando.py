import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

def visualize_prediction_histogram(predictions, class_names, image_index=0):
    """
    Visualizza un istogramma delle probabilità di previsione per una singola immagine.

    Args:
        predictions: Array di probabilità di previsione.
        class_names: Elenco dei nomi delle classi.
        image_index: Indice dell'immagine di test da visualizzare.
    """
    prediction = predictions[image_index]
    plt.figure(figsize=(8, 6))
    plt.bar(class_names, prediction)
    plt.xlabel("Classi")
    plt.ylabel("Probabilità")
    plt.title(f"Probabilità di Previsione per l'Immagine {image_index}")
    plt.xticks(rotation=45, ha="right")  # Rotazione delle etichette per leggibilità
    plt.tight_layout()
    plt.show()




print(tf.__version__)

category = ["T-shirt/top", "Trouser","Pullover","Dress", "Coat",
             "Sandal","Shirt","Sneakers","Bag", "ankle boot"]

categoria = ["Maglietta", "Pantaloni","Maglione","Vestito_donna", "Cappotto",
             "Sandalo","Camicia","Scarpe_ginnastica","borsa_donna", "stivale"]

# Carica il dataset Fashion MNIST
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()



# Normalizza le immagini (scala i valori dei pixel tra 0 e 1)
#train_images = train_images / 255.0
#test_images = test_images / 255.0

# Definisci il modello della rete neurale
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Appiattisce le immagini 28x28 in un vettore 1D
    keras.layers.Dense(128, activation='relu'),  # Strato denso con 128 nodi e funzione di attivazione ReLU
    keras.layers.Dense(10, activation='softmax') # Strato di output con 10 nodi (categorie) e funzione softmax
])

# Compila il modello
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Addestra il modello
model.fit(train_images, train_labels, epochs=5)#da 10 epoche a due

# Valuta il modello sul set di test
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\nAccuratezza sul test set:', test_acc)

index = np.random.randint(0, len(test_images))
print("Numero di immagini di test",len(test_images))

predictions = model.predict(test_images)
print(predictions[index])

visualize_prediction_histogram(predictions, category, image_index=index)

print("numero casuale", index)
n_indumento_trovato = np.argmax(predictions[index]) #12 sbaglia

print("categoria", n_indumento_trovato)
print(categoria[n_indumento_trovato])
n_indumento_reale = test_labels[index]
print(category[n_indumento_reale])

image = test_images[index]
label = test_labels[index]

# Visualizza l'immagine
plt.imshow(image, cmap='gray')
plt.title(f"Etichetta: {category[label]}")
plt.show()

