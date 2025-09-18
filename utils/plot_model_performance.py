import matplotlib.pyplot as plt

def plot_model_performance(train_loss, val_loss):

    """
    Grafica la pérdida de entrenamiento y validación a lo largo de las épocas.

    Args:
        train_loss (list): Una lista de valores de pérdida por época durante el entrenamiento.
        val_loss (list): Una lista de valores de pérdida por época durante la validación.
    """
    # Crear una lista de épocas basada en la longitud de los datos
    epocas = range(1, len(train_loss) + 1)
    
    # Configurar el gráfico
    plt.figure(figsize=(10, 6))  # Tamaño del gráfico
    plt.plot(epocas, train_loss, 'r-', label='Pérdida en Entrenamiento') # 'r-' para línea roja
    plt.plot(epocas, val_loss, 'b--', label='Pérdida en Validación') # 'b--' para línea azul discontinua
    
    # Añadir títulos y etiquetas
    plt.title('Comparación de Pérdida: Entrenamiento vs. Validación', fontsize=16)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Pérdida', fontsize=12)
    plt.legend()  # Mostrar la leyenda
    plt.grid(True) # Añadir cuadrícula para mejor lectura
    
    # Mostrar el gráfico
    plt.show()
