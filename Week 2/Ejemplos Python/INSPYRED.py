# -*- coding: utf-8 -*-
"""
@author: MARCELOFGB
"""
import random
import time
#import inspyred
from inspyred import ec
from inspyred.ec import variators
from inspyred.ec import selectors
from inspyred.ec import replacers

# --- 1. Definición del Problema y Funciones Básicas ---

# Parámetros del problema
NUM_BITS_PROBLEMA = 20  # La longitud de la cadena binaria (cromosoma) que queremos optimizar

def generador_binario(random, args):
    """
    Función generadora de individuos para la población inicial.
    Crea una lista de NUM_BITS_PROBLEMA con 0s y 1s aleatorios.
    """
    return [random.randint(0, 1) for _ in range(NUM_BITS_PROBLEMA)]

def evaluador_suma_bits(candidates, args):
    """
    Función de evaluación de la aptitud (fitness).
    Para cada candidato (individuo), calcula la suma de sus bits.
    El objetivo es maximizar esta suma.
    """
    fitness = []
    for candidate in candidates:
        fitness.append(sum(candidate)) # La aptitud es simplemente la suma de los 1s
    return fitness

def observador_custom(population, num_generations, num_evaluations, args):
    """
    Un observador personalizado para imprimir el progreso de cada generación.
    Muestra la mejor aptitud y el mejor individuo de la generación actual.
    """
    best_individual = max(population, key=lambda x: x.fitness)
    # Convertir el individuo candidato a una cadena para una mejor visualización
    best_candidate_str = "".join(map(str, best_individual.candidate))
    print(f"Generación: {num_generations:3d} | Mejor Aptitud: {best_individual.fitness:2d}/{NUM_BITS_PROBLEMA} | Mejor Individuo: {best_candidate_str}")

# --- 2. Configuración y Ejecución del Algoritmo Genético ---

def main():
    # Establecer la semilla aleatoria para reproducibilidad
    # Si quieres resultados diferentes cada vez, comenta o cambia esta línea.
    seed = int(time.time())
    random_state = random.Random(seed)
    print(f"Semilla aleatoria utilizada: {seed}")

    # Inicializar el motor del algoritmo genético de inspyred
    # ec.GA() es la clase para Algoritmos Genéticos
    ea = ec.GA(random_state)

    # --- 2.1. Componentes del GA ---

    # Selector: Cómo se eligen los individuos para el cruce.
    # tournament_selection es una de las más comunes y robustas.
    # El tamaño del torneo (tournament_size) es un parámetro clave aquí.
    ea.selector = selectors.tournament_selection

    # Variadores: Operaciones que crean nuevos individuos (cruce y mutación).
    # Se pueden encadenar varios variadores en una lista.
    # uniform_crossover: Intercambia bits entre dos padres con cierta probabilidad.
    # bit_flip_mutation: Invierte bits (0 a 1, 1 a 0) con cierta probabilidad.
    ea.variator = [
        variators.uniform_crossover,   # Cruce uniforme
        variators.bit_flip_mutation    # Mutación de bit
    ]

    # Reemplazador: Cómo se forma la nueva generación.
    # generational_replacement: La nueva generación reemplaza completamente a la anterior (elites pueden ser mantenidos).
    ea.replacer = replacers.generational_replacement

    # Observadores: Funciones que se ejecutan después de cada generación.
    # Usamos nuestro observador personalizado para ver el progreso.
    ea.observer = [observador_custom] # Puedes añadir inspyred.ec.observers.best_ever_observer también


    # --- 2.2. Parámetros de Ejecución del GA ---
    pop_size = 10000                 # Tamaño de la población (número de individuos)
    num_generations = 100          # Número máximo de generaciones a evolucionar
    num_elites = 1                 # Número de los mejores individuos que pasan directamente a la siguiente generación
    crossover_rate = 0.9           # Probabilidad de que se realice un cruce entre dos padres
    mutation_rate = 0.5           # Probabilidad de que un bit mute

    tournament_size = 3            # Tamaño del torneo para la selección

    # Configurar si el problema es de maximización o minimización
    maximize = True # Estamos maximizando la suma de bits (queremos todos 1s)

    print("\n--- Iniciando la Evolución del Algoritmo Genético ---")
    print(f"Número de bits por individuo: {NUM_BITS_PROBLEMA}")
    print(f"Tamaño de la población inicial: {pop_size}")
    print(f"Número máximo de generaciones: {num_generations}")
    print(f"Élites por generación (pasan directamente): {num_elites}")
    print(f"Tasa de cruce: {crossover_rate}")
    print(f"Tasa de mutación: {mutation_rate}")
    print(f"Tamaño del torneo de selección: {tournament_size}\n")


    # --- 2.3. Ejecutar el proceso de evolución ---
    # La función 'evolve' es el corazón de inspyred y ejecuta el GA.
    # Recibe:
    # - generator: La función para crear individuos iniciales.
    # - evaluator: La función para calcular la aptitud.
    # - pop_size: Tamaño de la población.
    # - num_generations: N° máximo de generaciones.
    # - maximize: Si es un problema de maximización.
    # - num_elites: N° de élites.
    # - crossover_rate, mutation_rate, tournament_size: Parámetros para los variadores y selector.
    #   Estos parámetros se pasan a los variadores y selectores configurados en 'ea'.
    final_population = ea.evolve(
        generator=generador_binario,
        evaluator=evaluador_suma_bits,
        pop_size=pop_size,
        num_generations=num_generations,
        maximize=maximize,
        num_elites=num_elites,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size  # Pasado al selector
    )

    # --- 3. Mostrar los Resultados Finales ---
    print("\n--- Evolución Finalizada ---")
    best_of_run = max(final_population, key=lambda x: x.fitness)
    best_candidate_str = "".join(map(str, best_of_run.candidate))
    print(f"\nMejor individuo encontrado: {best_candidate_str}")
    print(f"Aptitud final del mejor individuo: {best_of_run.fitness}/{NUM_BITS_PROBLEMA}")

    if best_of_run.fitness == NUM_BITS_PROBLEMA:
        print("¡El algoritmo genético encontró la solución óptima (todos 1s)!")
    else:
        print("El algoritmo genético no encontró la solución óptima en el número de generaciones dado.")

if __name__ == "__main__":
    main()