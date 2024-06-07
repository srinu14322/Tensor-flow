import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam

# Function to load datasets
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, header=None)

def app():
    st.sidebar.title("Configuration")
    
    # Number of Hidden Layers
    num_hidden_layers = st.sidebar.slider("Number of Hidden Layers", 1, 5, 1)
    hidden_layers = []

    for i in range(num_hidden_layers):
        st.sidebar.markdown(f"### Hidden Layer {i+1}")
        units = st.sidebar.slider(f"Number of units for hidden layer {i+1}", 1, 10, 1)
        activation = st.sidebar.selectbox(f"Activation function for hidden layer {i+1}", ['tanh', 'sigmoid','relu'])
        hidden_layers.append((units, activation))

    # Epoch slider
    epochs = st.sidebar.slider("Number of Epochs", 1, 500, 100)
    batch_size=st.sidebar.slider("batch size",1,100,1)

    # Dataset selection
    dataset_options = {
        "Dataset 1": "1.ushape.csv",
        "Dataset 2": "2.concerticcir1.csv",
        "Dataset 3": "3.concertriccir2.csv",
        "Dataset 4": "4.linearsep.csv",
        "Dataset 5": "5.outlier.csv",
        "Dataset 6": "6.overlap.csv",
        "Dataset 7": "7.xor.csv",
        "Dataset 8": "8.twospirals.csv"
    }
    dataset_choice = st.sidebar.selectbox("Choose a dataset", list(dataset_options.keys()))
    dataset_path = dataset_options[dataset_choice]

    if st.sidebar.button("Submit"):
        # Load dataset
        dataset = load_data(dataset_path)

        # Split dataset into training and testing sets
        X = dataset.iloc[:, :-1].values
        Y = dataset.iloc[:, -1].values.astype(np.int_)  # Convert target variable to integers

        # Define the input layer
        input_layer = Input(shape=(2,))

        def create_layer(units, activation):
            return Dense(units=units, activation=activation, use_bias=True)

        def build_model():
            layers = []
            for units, activation in hidden_layers:
                layers.append(create_layer(units, activation))
            return layers

        def compile_model(input_layer, layers):
            x = input_layer
            for layer in layers:
                x = layer(x)
            output_layer = Dense(1, activation='sigmoid')(x)  # Assuming binary classification
            model = Model(inputs=input_layer, outputs=output_layer)
            model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
            return model

        # Build and compile the model
        model_layers = build_model()
        model = compile_model(input_layer, model_layers)

        # Train the model
        model.fit(X, Y,batch_size= batch_size, epochs=epochs, verbose=0)

        # Get all hidden layers
        hidden_layers_output = [layer.output for layer in model.layers if isinstance(layer, Dense)]
        
        # Extract the output of each neuron from all hidden layers
        for layer_num, layer_output in enumerate(hidden_layers_output):
            num_neurons = layer_output.shape[1]
            for neuron_num in range(num_neurons):
                neuron_output = layer_output[:, neuron_num]
                neuron_model = Model(inputs=model.input, outputs=neuron_output)
                st.write(f"Plotting decision region for neuron {neuron_num+1} in hidden layer {layer_num+1}")
                fig, ax = plt.subplots()
                plot_decision_regions(X, Y, clf=neuron_model, ax=ax)
                st.pyplot(fig)

if __name__ == "__main__":
    app()
