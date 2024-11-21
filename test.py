import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random
from targets_plot_generator import generate_plot, image_addresses
# ---------------------------
# Data Preprocessing
# ---------------------------

def preprocessing():
    pass

# Load and preprocess data
df = pd.read_csv("TSLA_data.csv")
print(df.head())
df.set_index('DCP', inplace=True) 
features = df[['DNCP', 'OPCP', 'HPCP', 'LPCP', 'CPCP', 'ACPCP', 'VTCP']]
target = df['MPN5P']

scaler_features = MinMaxScaler().fit(features)
scaler_target = MinMaxScaler().fit(target.values.reshape(-1, 1))

features_scaled = scaler_features.transform(features)
target_scaled = scaler_target.transform(target.values.reshape(-1, 1))

def build_data_sequences(data_X, data_Y, timesteps):
    X, y = [], []
    for i in range(len(data_X) - timesteps):
        X.append(data_X[i:(i + timesteps), :])
        y.append(data_Y[i + timesteps])
    return np.array(X), np.array(y)

timesteps = 10  # Number of days to look back for each sequence
X, y = build_data_sequences(features_scaled, target_scaled, timesteps)
print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

# Split data into training and testing
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Genetic Algorithm Setup
# ---------------------------

# Hyperparameter space
hyperparameter_space = {
    "units_layer1": {"type": "int", "min": 32, "max": 256},
    "units_layer2": {"type": "int", "min": 32, "max": 256},
    "units_layer3": {"type": "int", "min": 32, "max": 256},
    "units_layer4": {"type": "int", "min": 32, "max": 256},
    "learning_rate": {"type": "float", "min": 1e-4, "max": 1e-2},
    "batch_size": {"type": "int", "choices": [16, 32, 64]}
}

# GA Parameters
population_size = 10
generations = 20
mutation_rate = 0.2
crossover_rate = 0.8
elitism = True  # Whether to keep the best individual

# Define chromosome as a dictionary of hyperparameters
def create_individual():
    individual = {}
    for param, param_info in hyperparameter_space.items():
        if param_info["type"] == "int":
            if "choices" in param_info:
                individual[param] = random.choice(param_info["choices"])
            else:
                individual[param] = random.randint(param_info["min"], param_info["max"])
        elif param_info["type"] == "float":
            individual[param] = random.uniform(param_info["min"], param_info["max"])
    return individual

def initialize_population(size):
    return [create_individual() for _ in range(size)]

# Define the fitness function
def fitness(individual):
    # Build and compile the model based on individual hyperparameters
    units1 = individual["units_layer1"]
    units2 = individual["units_layer2"]
    units3 = individual["units_layer3"]
    units4 = individual["units_layer4"]
    learning_rate = individual["learning_rate"]
    batch_size = individual["batch_size"]
    
    # Clear previous models from memory
    tf.keras.backend.clear_session()
    
    # Build the model
    model = Sequential([
        LSTM(units=units1, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        LSTM(units=units2, return_sequences=True),
        LSTM(units=units3, return_sequences=True),
        LSTM(units=units4),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='mse')
    
    # Early stopping to prevent long training times
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=30,  # Reduced epochs for speed; adjust as needed
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=0  # Suppress output for cleaner logs
    )
    
    # Evaluate the model on validation data
    val_loss = model.evaluate(X_val, y_val, verbose=0)
    
    # Since GA maximizes fitness, invert the loss
    fitness_score = -val_loss
    print(f"Evaluated Individual: {individual}, Validation Loss: {val_loss}, Fitness: {fitness_score}")
    return fitness_score

# Selection (Tournament Selection)
def selection(population, fitness_scores, k=3):
    selected = []
    for _ in range(2):  # Select two parents
        aspirants = random.sample(list(zip(population, fitness_scores)), k)
        aspirants = sorted(aspirants, key=lambda x: x[1], reverse=True)
        selected.append(aspirants[0][0])
    return selected

# Crossover (Uniform Crossover)
def crossover(parent1, parent2):
    child1, child2 = {}, {}
    for param in hyperparameter_space.keys():
        if random.random() < 0.5:
            child1[param] = parent1[param]
            child2[param] = parent2[param]
        else:
            child1[param] = parent2[param]
            child2[param] = parent1[param]
    return child1, child2

# Mutation
def mutate(individual):
    for param, param_info in hyperparameter_space.items():
        if random.random() < mutation_rate:
            if param_info["type"] == "int":
                if "choices" in param_info:
                    individual[param] = random.choice(param_info["choices"])
                else:
                    individual[param] = random.randint(param_info["min"], param_info["max"])
            elif param_info["type"] == "float":
                individual[param] = random.uniform(param_info["min"], param_info["max"])
    return individual

# ---------------------------
# Genetic Algorithm Execution
# ---------------------------

# Initialize population
population = initialize_population(population_size)

for generation in range(generations):
    print(f"\n=== Generation {generation + 1} ===")
    
    # Evaluate fitness for each individual
    fitness_scores = [fitness(ind) for ind in population]
    
    # Sort population based on fitness
    population_fitness = list(zip(population, fitness_scores))
    population_fitness.sort(key=lambda x: x[1], reverse=True)
    
    # Display best fitness in current generation
    best_individual, best_fitness = population_fitness[0]
    print(f"Best Fitness: {best_fitness} with hyperparameters: {best_individual}")
    
    # Elitism: carry forward the best individual to the next generation
    if elitism:
        new_population = [best_individual]
    else:
        new_population = []
    
    # Generate new individuals until the new population is full
    while len(new_population) < population_size:
        # Selection
        parent1, parent2 = selection(population, fitness_scores)
        
        # Crossover
        if random.random() < crossover_rate:
            child1, child2 = crossover(parent1, parent2)
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        
        # Mutation
        child1 = mutate(child1)
        child2 = mutate(child2)
        
        new_population.extend([child1, child2])
    
    # Ensure the population size remains constant
    population = new_population[:population_size]

# ---------------------------
# Final Evaluation
# ---------------------------

# After all generations, evaluate and select the best individual
final_fitness_scores = [fitness(ind) for ind in population]
final_population_fitness = list(zip(population, final_fitness_scores))
final_population_fitness.sort(key=lambda x: x[1], reverse=True)

best_individual, best_fitness = final_population_fitness[0]
print(f"\n=== Best Hyperparameters Found ===")
print(f"Hyperparameters: {best_individual}")
print(f"Validation Fitness (Negative Loss): {best_fitness}")

# Optional: Retrain the model with the best hyperparameters on the entire training set
# and evaluate on the test set if you have one.

# ---------------------------
# Making Predictions with the Best Model
# ---------------------------

# Assuming you have a separate test set, you can build and train the final model here
# For demonstration, we'll use the existing X_val and y_val as test data

def build_final_model(best_individual):
    units1 = best_individual["units_layer1"]
    units2 = best_individual["units_layer2"]
    units3 = best_individual["units_layer3"]
    units4 = best_individual["units_layer4"]
    learning_rate = best_individual["learning_rate"]
    batch_size = best_individual["batch_size"]
    
    tf.keras.backend.clear_session()
    
    model = Sequential([
        LSTM(units=units1, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        LSTM(units=units2, return_sequences=True),
        LSTM(units=units3, return_sequences=True),
        LSTM(units=units4),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=0
    )
    
    return model

# Build and train the final model
final_model = build_final_model(best_individual)

# Evaluate on validation data
final_val_loss = final_model.evaluate(X_val, y_val, verbose=0)
print(f"Final Validation Loss: {final_val_loss}")

# At the end of the script, after GA optimization:
final_model = build_final_model(best_individual)

# Make predictions
predictions = final_model.predict(X_val)
predictions = scaler_target.inverse_transform(predictions)
y_true = scaler_target.inverse_transform(y_val)

# Use dates from the original dataset for visualization
dates = df.index[-len(predictions):]  # Assuming the last 'len(predictions)' dates match

# Specify the target variable's name for plotting (e.g., "Stock Price")
target_name = "Stock Price"

# Generate plots
plot_results = generate_plot(predictions, y_true, dates, target_name)

# Display results
print("\n=== Plot Results ===")
for plot_type, metrics in plot_results.items():
    if isinstance(metrics, dict):
        print(f"{plot_type}: {metrics}")
    else:
        print(f"{plot_type} saved at: {metrics}")