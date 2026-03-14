# -*- coding: utf-8 -*-
"""
Created on Thu May  8 23:54:50 2025

@author: MARCELOFGB
"""

import random

# --- Parámetros del Algoritmo CHC y del Problema ---
POP_SIZE = 700  # Tamaño de la población (N)
GENE_LENGTH = 25 # Longitud del cromosoma (L)
TARGET_STRING = "10" * GENE_LENGTH # Objetivo: una cadena de todos '1's
MAX_GENERATIONS = 10 # Número máximo de generaciones

# Parámetros específicos de CHC
# Umbral inicial de divergencia (d). A menudo se inicializa a L/4.
# Se irá decrementando si no se producen cruces.
INITIAL_D_THRESHOLD_DIVISOR = 4
# Tasa de mutación para el reinicio cataclísmico (cuando la población converge)
# Es un porcentaje de bits a cambiar respecto al mejor individuo.
# Eshelman sugiere ~35%
RESTART_MUTATION_RATE = 0.35

# --- Funciones Auxiliares ---

def create_individual():
    """Crea un individuo aleatorio (cadena de bits)."""
    return "".join(random.choice("01") for _ in range(GENE_LENGTH))

def initialize_population():
    """Inicializa la población con individuos aleatorios."""
    population = []
    for _ in range(POP_SIZE):
        chromosome = create_individual()
        population.append({"chromosome": chromosome, "fitness": calculate_fitness(chromosome)})
    return population

def calculate_fitness(chromosome):
    """Calcula la aptitud: número de bits que coinciden con TARGET_STRING."""
    return sum(1 for i in range(GENE_LENGTH) if chromosome[i] == TARGET_STRING[i])

def hamming_distance(str1, str2):
    """Calcula la distancia de Hamming entre dos cadenas de la misma longitud."""
    distance = 0
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            distance += 1
    return distance

def hux_crossover(parent1_chrom, parent2_chrom):
    """
    Operador de Cruce HUX (Half Uniform Crossover).
    Intercambia exactamente la mitad de los bits en los que los padres difieren.
    Esto maximiza la distancia genética entre los padres y los hijos,
    y entre los dos hijos.
    """
    diff_indices = [i for i, (b1, b2) in enumerate(zip(parent1_chrom, parent2_chrom)) if b1 != b2]
    
    # Si no hay diferencias, los hijos son clones de los padres (improbable con el umbral d)
    if not diff_indices:
        return parent1_chrom, parent2_chrom

    # Barajamos los índices donde difieren
    random.shuffle(diff_indices)
    
    # Tomamos la mitad de estos índices para intercambiar
    # Si el número de diferencias es impar, //2 redondeará hacia abajo.
    num_to_swap = len(diff_indices) // 2
    
    offspring1_list = list(parent1_chrom)
    offspring2_list = list(parent2_chrom)

    for i in range(num_to_swap):
        idx = diff_indices[i]
        # Intercambiamos los bits en esta posición entre los padres para formar los hijos
        offspring1_list[idx] = parent2_chrom[idx]
        offspring2_list[idx] = parent1_chrom[idx]
        
    return "".join(offspring1_list), "".join(offspring2_list)

# --- Algoritmo CHC ---

def chc_algorithm():
    print("Iniciando Algoritmo CHC...")
    print(f"Tamaño Población: {POP_SIZE}, Longitud Gen: {GENE_LENGTH}")
    print(f"Objetivo: {TARGET_STRING}\n")

    # 1. Inicialización
    population = initialize_population()
    # Ordenar la población inicial por fitness (descendente)
    population.sort(key=lambda ind: ind["fitness"], reverse=True)

    best_overall_individual = population[0].copy()
    
    # Umbral de divergencia 'd'
    # Inicialmente, d puede ser L/4 o L/2. Se decrementa si no se producen cruces.
    # El cruce solo ocurre si la distancia Hamming entre padres > d.
    current_d_threshold = GENE_LENGTH // INITIAL_D_THRESHOLD_DIVISOR 
    
    print(f"Umbral 'd' inicial: {current_d_threshold}")

    for generation in range(MAX_GENERATIONS):
        print(f"\n--- Generación {generation + 1} ---")
        print(f"Mejor fitness actual: {population[0]['fitness']}, d_threshold: {current_d_threshold}")

        # Selección de padres: se baraja la población actual (P) y se emparejan.
        # En CHC, la población de padres (P_padres) es la misma que la población actual (P_t).
        parent_pool_indices = list(range(POP_SIZE))
        random.shuffle(parent_pool_indices)
        
        offspring_population = [] # P_hijos

        # 2. Recombinación Heterogénea (HUX)
        # Se itera sobre pares de padres seleccionados aleatoriamente de la población actual.
        num_crossovers_done = 0
        for i in range(0, POP_SIZE - 1, 2): # Emparejamiento
            idx1, idx2 = parent_pool_indices[i], parent_pool_indices[i+1]
            parent1 = population[idx1]
            parent2 = population[idx2]

            dist = hamming_distance(parent1["chromosome"], parent2["chromosome"])

            # Solo se cruzan si la distancia de Hamming es mayor que el umbral 'd'
            if dist > current_d_threshold:
                num_crossovers_done += 1
                offspring1_chrom, offspring2_chrom = hux_crossover(parent1["chromosome"], parent2["chromosome"])
                
                # Evaluar nuevos hijos
                offspring_population.append({
                    "chromosome": offspring1_chrom,
                    "fitness": calculate_fitness(offspring1_chrom)
                })
                offspring_population.append({
                    "chromosome": offspring2_chrom,
                    "fitness": calculate_fitness(offspring2_chrom)
                })

        print(f"Cruces realizados en esta generación: {num_crossovers_done}")

        # 3. Selección Conservativa (Cross-generational elitist selection)
        # Se combinan los padres (P_t) y los hijos (P_hijos)
        # Se seleccionan los N mejores individuos para formar la nueva población (P_t+1)
        combined_population = population + offspring_population
        combined_population.sort(key=lambda ind: ind["fitness"], reverse=True)
        
        new_population = []
        # Asegurarse de no tener duplicados (CHC a menudo lo hace implícitamente por HUX)
        # Para este ejemplo didáctico, una simple selección de los mejores N es suficiente.
        # CHC original puede tener lógicas más complejas para evitar duplicados exactos.
        added_chromosomes = set()
        for ind in combined_population:
            if len(new_population) < POP_SIZE:
                # Opcional: evitar duplicados exactos si se desea ser más estricto
                # if ind["chromosome"] not in added_chromosomes: 
                new_population.append(ind)
                #    added_chromosomes.add(ind["chromosome"])
            else:
                break
        
        population = new_population

        # Actualizar el mejor individuo global si es necesario
        if population[0]["fitness"] > best_overall_individual["fitness"]:
            best_overall_individual = population[0].copy()
            print(f"¡Nuevo mejor global encontrado! Fitness: {best_overall_individual['fitness']}, Individuo: {best_overall_individual['chromosome']}")

        # Condición de parada: objetivo encontrado
        if best_overall_individual["fitness"] == GENE_LENGTH:
            print(f"\n¡Objetivo encontrado en la generación {generation + 1}!")
            break

        # 4. Gestión del umbral 'd' y Mutación Cataclísmica (Reinicio)
        
        # Si no se generaron hijos (num_crossovers_done == 0),
        # significa que la población se ha vuelto demasiado similar (todas las distancias <= d).
        # En este caso, se decrementa 'd'.
        if num_crossovers_done == 0 and len(offspring_population) == 0 : # Asegura que no hubo cruces
            current_d_threshold -= 1
            print(f"No hubo cruces. Umbral 'd' decrementado a: {current_d_threshold}")

        # Si 'd' cae por debajo de 0 (o un valor muy bajo),
        # la población ha convergido y se necesita un reinicio (mutación cataclísmica).
        if current_d_threshold < 0:
            print("\n*** ¡REINICIO POR MUTACIÓN CATACLÍSMICA! ***")
            
            # Conservar el mejor individuo encontrado hasta ahora.
            elite_individual = best_overall_individual.copy() # Usar el mejor global
            
            new_population_after_restart = [elite_individual]
            
            # Generar el resto de la población (N-1 individuos)
            # mutando copias del mejor individuo.
            # La mutación es un porcentaje de bits (RESTART_MUTATION_RATE)
            # que se cambian respecto al cromosoma del mejor.
            num_bits_to_mutate = int(GENE_LENGTH * RESTART_MUTATION_RATE)
            if num_bits_to_mutate == 0 and GENE_LENGTH > 0 : num_bits_to_mutate = 1 # asegurar al menos 1 bit si es posible

            for _ in range(POP_SIZE - 1):
                mutant_chrom_list = list(elite_individual["chromosome"])
                
                # Seleccionar 'num_bits_to_mutate' posiciones aleatorias para cambiar
                indices_to_mutate = random.sample(range(GENE_LENGTH), num_bits_to_mutate)
                
                for idx in indices_to_mutate:
                    mutant_chrom_list[idx] = '1' if mutant_chrom_list[idx] == '0' else '0'
                
                mutant_chrom = "".join(mutant_chrom_list)
                new_population_after_restart.append({
                    "chromosome": mutant_chrom,
                    "fitness": calculate_fitness(mutant_chrom)
                })
            
            population = new_population_after_restart
            population.sort(key=lambda ind: ind["fitness"], reverse=True) # Re-ordenar

            # Reiniciar el umbral 'd' a su valor inicial (o un valor basado en la nueva diversidad)
            current_d_threshold = GENE_LENGTH // INITIAL_D_THRESHOLD_DIVISOR
            print(f"Población reiniciada. Mejor individuo conservado. Nuevo umbral 'd': {current_d_threshold}")

    # --- Fin del Algoritmo ---
    print("\n--- Simulación Finalizada ---")
    print(f"Mejor individuo encontrado: {best_overall_individual['chromosome']}")
    print(f"Fitness: {best_overall_individual['fitness']} / {GENE_LENGTH}")
    if best_overall_individual['fitness'] < GENE_LENGTH:
        print(f"Se alcanzó el máximo de generaciones ({MAX_GENERATIONS}).")

# --- Ejecutar el algoritmo ---
if __name__ == "__main__":
    random.seed(42) # Para reproducibilidad del ejemplo
    chc_algorithm()