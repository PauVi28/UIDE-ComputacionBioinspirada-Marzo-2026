# -*- coding: utf-8 -*-
"""
Created on Fri May  9 02:13:32 2025

@author: MARCELOFGB
"""

import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Representación del Árbol ---
class Node:
    """Clase base para los nodos del árbol."""
    def evaluate(self, x_val):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def get_nodes_and_depth(self, current_depth=0):
        """Devuelve una lista de (nodo, profundidad) para este nodo y sus descendientes."""
        nodes = [(self, current_depth)]
        if hasattr(self, 'children'):
            for child in self.children:
                nodes.extend(child.get_nodes_and_depth(current_depth + 1))
        return nodes
    
    def get_node_count(self):
        """Cuenta el número total de nodos en el subárbol."""
        count = 1
        if hasattr(self, 'children'):
            for child in self.children:
                count += child.get_node_count()
        return count


class FunctionNode(Node):
    """Nodo que representa una función (operador)."""
    def __init__(self, func, func_name, children):
        self.func = func
        self.func_name = func_name
        self.children = children # Espera una lista de nodos hijos

    def evaluate(self, x_val):
        evaluated_children = [child.evaluate(x_val) for child in self.children]
        try:
            return self.func(*evaluated_children)
        except ZeroDivisionError:
            return 1.0 # Protección contra división por cero
        except OverflowError:
            return 1.0e6 # Protección contra overflow
        except ValueError: # math domain error
            return 1.0

    def __str__(self):
        child_strs = ", ".join(str(c) for c in self.children)
        return f"{self.func_name}({child_strs})"

class TerminalNode(Node):
    """Nodo que representa un terminal (variable o constante)."""
    def __init__(self, value):
        self.value = value # Puede ser 'x' o un número

    def evaluate(self, x_val):
        if self.value == 'x':
            return x_val
        return float(self.value)

    def __str__(self):
        return str(self.value)

# --- Conjunto de Funciones y Terminales ---
FUNCTIONS = {
    '+': (lambda a, b: a + b, 2), # función, aridad (número de argumentos)
    '-': (lambda a, b: a - b, 2),
    '*': (lambda a, b: a * b, 2),
    # Se podría añadir división protegida, sin, cos, etc.
    # 'p_div': (lambda a, b: a / b if b != 0 else 1.0, 2)
}
TERMINALS = ['x'] + [str(i) for i in range(-5, 6)] # 'x' y constantes de -5 a 5
# TERMINALS = ['x'] + [f"{random.uniform(-5, 5):.2f}" for _ in range(10)] # 'x' y constantes flotantes


# --- 2. Creación de Árboles Aleatorios (Inicialización) ---
def create_random_tree(max_depth, current_depth=0, method="grow"):
    """
    Crea un árbol aleatorio.
    Método 'grow': puede elegir un terminal antes de la max_depth.
    Método 'full': siempre elige funciones hasta max_depth.
    """
    if current_depth >= max_depth or (method == "grow" and random.random() < 0.4): # 0.4 es prob de ser terminal
        # Crear un nodo terminal
        term = random.choice(TERMINALS)
        return TerminalNode(term)
    else:
        # Crear un nodo de función
        func_name = random.choice(list(FUNCTIONS.keys()))
        func, arity = FUNCTIONS[func_name]
        children = [create_random_tree(max_depth, current_depth + 1, method) for _ in range(arity)]
        return FunctionNode(func, func_name, children)

# --- 3. Función de Fitness ---
# Datos de entrenamiento (para y = x^2 + x + 1)
X_TRAIN = np.linspace(-5, 5, 20)
Y_TARGET = X_TRAIN**2 + X_TRAIN + 1

def calculate_fitness(individual_tree):
    """Calcula el fitness como el Error Cuadrático Medio (ECM)."""
    mse = 0
    for x_val, y_true in zip(X_TRAIN, Y_TARGET):
        try:
            y_pred = individual_tree.evaluate(x_val)
            # Penalizar valores muy grandes que pueden surgir
            if abs(y_pred) > 1e6: # Umbral grande
                y_pred = 1e6 * np.sign(y_pred)
            mse += (y_pred - y_true)**2
        except (OverflowError, ValueError):
            mse += 1e12 # Penalización grande por errores numéricos
    
    # Queremos minimizar MSE, pero a menudo las funciones de fitness se maximizan.
    # Podemos devolver 1 / (1 + MSE) o simplemente el MSE y minimizarlo.
    # Para este ejemplo, devolveremos MSE directamente y buscaremos el mínimo.
    return mse / len(X_TRAIN)


# --- 4. Operadores Genéticos ---

# --- a. Selección ---
def tournament_selection(population_with_fitness, tournament_size):
    selected_tournament = random.sample(population_with_fitness, tournament_size)
    # Ordenar por fitness (menor MSE es mejor)
    selected_tournament.sort(key=lambda item: item[1])
    return selected_tournament[0][0] # Devuelve el árbol (individuo)

# --- b. Cruce (Crossover) ---
def get_random_node_and_parent(tree_root, parent=None, child_index=None):
    """
    Elige un nodo aleatorio del árbol y devuelve el nodo, su padre y el índice del hijo.
    Necesario para reemplazar el nodo durante el cruce/mutación.
    """
    nodes = [] # (node, parent, child_index_in_parent)
    
    # Usamos una pila para recorrido DFS no recursivo
    # (current_node, parent_of_current, index_of_current_in_parent_children)
    stack = [(tree_root, parent, child_index)]
    
    all_nodes_info = []

    while stack:
        curr_node, curr_parent, curr_child_idx = stack.pop()
        all_nodes_info.append((curr_node, curr_parent, curr_child_idx))
        
        if isinstance(curr_node, FunctionNode):
            for i, child in enumerate(curr_node.children):
                stack.append((child, curr_node, i))
                
    return random.choice(all_nodes_info)


def subtree_crossover(parent1_tree, parent2_tree):
    # Crear copias profundas para no modificar los originales
    child1 = copy.deepcopy(parent1_tree)
    child2 = copy.deepcopy(parent2_tree)

    # Punto de cruce en el hijo 1 (que es copia del padre 1)
    node1_to_replace, parent_of_node1, idx_in_parent1 = get_random_node_and_parent(child1)
    
    # Punto de cruce en el hijo 2 (que es copia del padre 2)
    subtree_from_p2, _, _ = get_random_node_and_parent(child2) # No necesitamos padre/índice del subárbol que se copia

    # Realizar el cruce en child1: reemplazar node1_to_replace con una copia de subtree_from_p2
    if parent_of_node1 is None: # El nodo a reemplazar es la raíz de child1
        child1 = copy.deepcopy(subtree_from_p2)
    else: # El nodo a reemplazar es un hijo de parent_of_node1
        parent_of_node1.children[idx_in_parent1] = copy.deepcopy(subtree_from_p2)

    # Ahora el cruce simétrico para child2 (usando el subárbol original de parent1)
    # Necesitamos obtener el subárbol original de parent1 que fue `node1_to_replace` antes de la copia.
    # Esto es más fácil si seleccionamos el nodo del `parent1_tree` original.
    node1_original_subtree, _, _ = get_random_node_and_parent(parent1_tree)
    
    node2_to_replace, parent_of_node2, idx_in_parent2 = get_random_node_and_parent(child2) # child2 ya fue modificado si P1 era su raíz
                                                                                         # pero el subárbol que vamos a reemplazar es aleatorio de child2

    if parent_of_node2 is None: # El nodo a reemplazar es la raíz de child2
        child2 = copy.deepcopy(node1_original_subtree)
    else:
        parent_of_node2.children[idx_in_parent2] = copy.deepcopy(node1_original_subtree)
        
    # Control de tamaño opcional (ej: si son muy grandes, devolver los padres)
    # if child1.get_node_count() > MAX_NODES_AFTER_OP: child1 = parent1_tree
    # if child2.get_node_count() > MAX_NODES_AFTER_OP: child2 = parent2_tree

    return child1, child2


# --- c. Mutación ---
def subtree_mutation(tree, max_depth_mutation_subtree=3):
    mutant = copy.deepcopy(tree)
    
    node_to_mutate, parent_of_node, idx_in_parent = get_random_node_and_parent(mutant)
    
    # Generar un nuevo subárbol aleatorio
    new_subtree = create_random_tree(max_depth=max_depth_mutation_subtree, method="grow")
    
    if parent_of_node is None: # El nodo a mutar es la raíz
        mutant = new_subtree
    else: # El nodo a mutar es un hijo
        parent_of_node.children[idx_in_parent] = new_subtree
        
    # Control de tamaño opcional
    # if mutant.get_node_count() > MAX_NODES_AFTER_OP: mutant = tree
    return mutant

# --- 5. Algoritmo Genético Principal ---
POPULATION_SIZE = 100
MAX_GENERATIONS = 5
TOURNAMENT_SIZE = 5
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.2
MAX_INITIAL_DEPTH = 4
MAX_DEPTH_MUTATION = 2 # Profundidad máxima para subárboles de mutación
ELITISM_COUNT = 2 # Cuántos mejores individuos pasan directamente

# --- Inicializar Población ---
population = [create_random_tree(MAX_INITIAL_DEPTH, method="ramped_half_and_half" if i < POPULATION_SIZE/2 else "grow") 
              for i in range(POPULATION_SIZE)] # Usar ramped half-and-half para diversidad inicial
                                               # (aquí simplificado a mitad grow, mitad full-like si se implementara 'full')

print("Ejemplo de Programación Genética para y = x^2 + x + 1")
print(f"Población: {POPULATION_SIZE}, Generaciones: {MAX_GENERATIONS}\n")

best_fitness_per_gen = []

for gen in range(MAX_GENERATIONS):
    # Evaluar fitness de la población actual
    population_with_fitness = []
    for ind_tree in population:
        fitness = calculate_fitness(ind_tree)
        population_with_fitness.append((ind_tree, fitness))

    # Ordenar por fitness (menor MSE es mejor)
    population_with_fitness.sort(key=lambda item: item[1])
    
    best_current_individual = population_with_fitness[0][0]
    best_current_fitness = population_with_fitness[0][1]
    best_fitness_per_gen.append(best_current_fitness)

    print(f"Generación {gen+1}: Mejor Fitness (MSE) = {best_current_fitness:.4f}")
    print(f"  Mejor Individuo: {str(best_current_individual)}")

    if best_current_fitness < 1e-3: # Criterio de parada temprano si es suficientemente bueno
        print("\nSolución suficientemente buena encontrada.")
        break

    # Crear nueva población
    new_population = []

    # Elitismo: los mejores individuos pasan directamente
    for i in range(ELITISM_COUNT):
        new_population.append(population_with_fitness[i][0])

    # Llenar el resto de la nueva población
    while len(new_population) < POPULATION_SIZE:
        parent1 = tournament_selection(population_with_fitness, TOURNAMENT_SIZE)
        
        if random.random() < CROSSOVER_RATE:
            parent2 = tournament_selection(population_with_fitness, TOURNAMENT_SIZE)
            child1, child2 = subtree_crossover(parent1, parent2)
            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)
        else: # Reproducción (o solo mutación)
            offspring = copy.deepcopy(parent1)
            new_population.append(offspring)


        # Aplicar mutación al último/últimos añadidos (si no se llenó ya la población)
        if len(new_population) > ELITISM_COUNT: # No mutar a los élites que ya pasaron
            idx_to_mutate = -1
            if len(new_population) % 2 == 0 and random.random() < CROSSOVER_RATE : # Si se añadieron 2 por crossover
                 if random.random() < MUTATION_RATE:
                    new_population[-2] = subtree_mutation(new_population[-2], MAX_DEPTH_MUTATION)
            
            if random.random() < MUTATION_RATE: # Mutar el último añadido
                 new_population[-1] = subtree_mutation(new_population[-1], MAX_DEPTH_MUTATION)


    population = new_population[:POPULATION_SIZE] # Asegurar tamaño exacto

# --- Resultados Finales ---
print("\n--- Evolución Terminada ---")
final_population_with_fitness = []
for ind_tree in population:
    fitness = calculate_fitness(ind_tree)
    final_population_with_fitness.append((ind_tree, fitness))
final_population_with_fitness.sort(key=lambda item: item[1])

best_overall_individual = final_population_with_fitness[0][0]
best_overall_fitness = final_population_with_fitness[0][1]

print(f"Mejor Fitness Global (MSE): {best_overall_fitness:.4f}")
print(f"Mejor Individuo Global: {str(best_overall_individual)}")
print(f"  (Target: y = x^2 + x + 1)")

# --- 6. Visualización ---
# Graficar evolución del fitness
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(best_fitness_per_gen) + 1), best_fitness_per_gen)
plt.title("Evolución del Mejor Fitness (MSE) por Generación")
plt.xlabel("Generación")
plt.ylabel("Mejor Fitness (MSE)")
plt.yscale('log') # Usar escala logarítmica si el fitness cambia mucho
plt.grid(True)

# Graficar la función encontrada vs la objetivo
y_predicted_best = [best_overall_individual.evaluate(x) for x in X_TRAIN]

plt.subplot(1, 2, 2)
plt.plot(X_TRAIN, Y_TARGET, 'bo-', label='Target (x^2 + x + 1)')
plt.plot(X_TRAIN, y_predicted_best, 'ro--', label=f'GP Encontrado: {str(best_overall_individual)}')
# Limitar el eje Y si las predicciones son muy dispares
max_abs_target = np.max(np.abs(Y_TARGET))
plt.ylim(-2 * max_abs_target -5 , 2 * max_abs_target + 5) # Ajusta el factor según sea necesario

plt.title("Función Objetivo vs. Función Encontrada por PG")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()
