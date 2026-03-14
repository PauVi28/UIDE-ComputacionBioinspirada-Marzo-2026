# -*- coding: utf-8 -*-
"""

@author: MARCELOFGB
"""

import random
import time
import numpy # Para las estadísticas (media, desviación estándar)

# Importaciones específicas de DEAP
from deap import base
from deap import creator
from deap import tools
#from deap import algorithms

# --- 1. Definición del Problema y Componentes Básicos ---

# Parámetros del problema
NUM_BITS_PROBLEMA = 20  # La longitud de la cadena binaria (cromosoma) que queremos optimizar

# DEAP requiere que definamos la clase de Fitness y la clase de Individuo.
# creator.create se encarga de crear dinámicamente estas clases.
# "FitnessMax": Es una clase de aptitud que indicamos que queremos maximizar.
#               Los pesos (1.0,) indican que es un problema de 1 objetivo y que ese objetivo se maximiza.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# "Individual": Es la clase de nuestro individuo, que será una lista basada en Python,
#               y tendrá un atributo 'fitness' que será una instancia de FitnessMax.
creator.create("Individual", list, fitness=creator.FitnessMax)

# --- Configuración del Toolbox de DEAP ---
# El Toolbox es la estructura central en DEAP donde registramos todas las funciones que usará el GA.
toolbox = base.Toolbox()

# Atributo genético: Cómo se crea un solo bit (0 o 1).
# Esto se usará para construir el individuo.
toolbox.register("attr_bool", random.randint, 0, 1)

# Estructura del individuo: Cómo se crea un individuo completo.
# tools.initRepeat: Crea un Individual repitiendo la función 'attr_bool' 'n' veces.
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=NUM_BITS_PROBLEMA)

# Estructura de la población: Cómo se crea una población.
# tools.initRepeat: Crea una lista (la población) repitiendo la función 'individual' 'pop_size' veces.
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Función de Evaluación (comparable a evaluador_suma_bits de inspyred)
def evalOneMax(individual):
    """
    Función de aptitud para el problema Ones Count (o OneMax).
    Calcula la suma de los bits de un individuo.
    DEAP espera un TUPLA de flotantes como retorno de la aptitud,
    incluso para un solo objetivo.
    """
    return sum(individual), # La coma es crucial para crear una tupla (e.g., (15,))

toolbox.register("evaluate", evalOneMax)

# Operador de Cruce (Crossover)
# cxUniform: Operador de cruce uniforme. indpb es la probabilidad de intercambiar CADA gen.
toolbox.register("mate", tools.cxUniform, indpb=0.5) # indpb = 0.5 es un valor común para cxUniform

# Operador de Mutación
# mutFlipBit: Mutación de bit. indpb es la probabilidad de que CADA bit mute.
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05) # Corresponde a mutation_rate = 0.05

# Operador de Selección
# selTournament: Selección por torneo. tournsize es el tamaño del torneo.
toolbox.register("select", tools.selTournament, tournsize=3) # Corresponde a tournament_size = 3

# --- 2. Configuración y Ejecución del Algoritmo Genético ---

def main():
    # Establecer la semilla aleatoria para reproducibilidad
    # Si quieres resultados diferentes cada vez, comenta o cambia esta línea.
    seed = int(time.time())
    random.seed(seed) # DEAP usa el módulo random estándar de Python
    print(f"Semilla aleatoria utilizada: {seed}")

    # Parámetros de Ejecución del GA (copiados del ejemplo de inspyred)
    pop_size = 1000                 # Tamaño de la población (número de individuos)
    num_generations = 100          # Número máximo de generaciones a evolucionar
    num_elites = 1                 # Número de los mejores individuos que pasan directamente a la siguiente generación
    crossover_prob = 0.9           # Probabilidad de que se realice un cruce (cxpb en DEAP)
    # mutation_prob ya está en toolbox.register("mutate", ..., indpb=0.05)
    # Sin embargo, algorithms.eaSimple también tiene un 'mutpb' que es la probabilidad
    # de que un INDIVIDUO mute. Aquí, lo estableceremos en 1.0 para que la indpb del
    # mutFlipBit sea la que controle la tasa de mutación real a nivel de bit.
    individual_mutation_prob = 0.98


    print("\n--- Iniciando la Evolución del Algoritmo Genético (DEAP) ---")
    print(f"Número de bits por individuo: {NUM_BITS_PROBLEMA}")
    print(f"Tamaño de la población inicial: {pop_size}")
    print(f"Número máximo de generaciones: {num_generations}")
    print(f"Élites por generación (pasan directamente): {num_elites}")
    print(f"Probabilidad de cruce (cxpb): {crossover_prob}")
    print(f"Probabilidad de mutación por bit (indpb): {toolbox.mutate.keywords['indpb']}") # Extraemos de la configuración
    print(f"Probabilidad de mutación por individuo (mutpb): {individual_mutation_prob}")
    print(f"Tamaño del torneo de selección: {toolbox.select.keywords['tournsize']}\n")

    # Crear la población inicial
    population = toolbox.population(n=pop_size)

    # Evaluar la aptitud de los individuos iniciales
    # La aptitud de cada individuo se almacena en su atributo .fitness.values
    # tools.map es una versión de map que funciona bien con DEAP
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Preparar las estadísticas para el logbook
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    # HallOfFame para guardar el mejor individuo encontrado hasta ahora (elites)
    hof = tools.HallOfFame(num_elites)

    # Logbook para registrar el progreso de la evolución
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "avg", "max", "std" # Definir la cabecera

    # Grabar las estadísticas de la población inicial (Generación 0)
    record = stats.compile(population)
    hof.update(population) # Actualizar el HallOfFame para la generación 0 también
    logbook.record(gen=0, evals=len(population), **record)
    # Simular el observador custom de inspyred para la primera generación
    best_gen0 = tools.selBest(population, 1)[0]
    best_candidate_str_gen0 = "".join(map(str, best_gen0))

    # CAMBIO AQUÍ: Formatear como flotante con 0 decimales
    print(f"Generación: {0:3d} | Mejor Aptitud: {best_gen0.fitness.values[0]:2.0f}/{NUM_BITS_PROBLEMA} | Mejor Individuo: {best_candidate_str_gen0}")


    # --- Bucle Principal del Algoritmo Genético ---
    # Iterar a través de las generaciones
    for gen in range(1, num_generations + 1):
        # Seleccionar los individuos para la próxima generación (Padres)
        offspring = toolbox.select(population, pop_size)
        # Clonar los individuos seleccionados para no modificar la población original
        offspring = list(map(toolbox.clone, offspring))

        # Aplicar Cruce y Mutación a los descendientes
        # Cruce: Se seleccionan pares, se cruzan si cumplen cxpb, y se invalidan sus aptitudes para reevaluarlas
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values # Invalida la aptitud para que sea reevaluada
                del child2.fitness.values # Invalida la aptitud para que sea reevaluada

        # Mutación: Se aplica mutación a cada individuo si cumple individual_mutation_prob
        for mutant in offspring:
            if random.random() < individual_mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values # Invalida la aptitud para que sea reevaluada

        # Evaluar los individuos con aptitudes inválidas (los nuevos mutantes y descendientes)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Elitismo: Mantener los mejores individuos de la generación anterior
        # Esta es una forma común de manejar el elitismo manualmente con DEAP.
        # Primero, se combinan la población actual con los nuevos descendientes.
        # Luego, se seleccionan los 'pop_size' mejores para la siguiente generación,
        # asegurando que los 'num_elites' mejores de la generación anterior (si son mejores que los nuevos)
        # tengan la oportunidad de ser transferidos.
        # Para implementar el elitismo directamente, podemos añadir los elites del hof a la descendencia
        # y luego seleccionar los pop_size mejores.
        
        # Copiamos la población actual para no modificarla directamente en la selección de élites
        current_pop_for_elitism = list(population) 
        
        # Los élites se toman del HallOfFame para asegurar que sean los mejores de todas las generaciones pasadas
        # hasta la actual.
        elites = [toolbox.clone(ind) for ind in hof] # Clonar para no modificar el hof original

        # La nueva población se forma a partir de la descendencia + los élites
        # y luego se seleccionan los pop_size mejores de este grupo combinado para formar la siguiente generación
        # (Esto es una implementación común de "replacement with elitism")
        population[:] = toolbox.select(offspring + elites, pop_size)


        # Actualizar el Hall Of Fame con la nueva población (ya contiene los mejores elites)
        # Se necesita `hof.update(population)` porque `population` ahora es la nueva generación
        # que podría incluir nuevos mejores o mantener los viejos.
        hof.update(population)

        # Grabar las estadísticas de la generación actual
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(invalid_ind), **record) # len(invalid_ind) es el número de nuevas evaluaciones

        # Simular el observador_custom de inspyred
        best_current_gen = tools.selBest(population, 1)[0]
        best_candidate_str_current_gen = "".join(map(str, best_current_gen))
        # CAMBIO AQUÍ: Formatear como flotante con 0 decimales
        print(f"Generación: {gen:3d} | Mejor Aptitud: {best_current_gen.fitness.values[0]:2.0f}/{NUM_BITS_PROBLEMA} | Mejor Individuo: {best_candidate_str_current_gen}")

        # Condición de parada temprana: Si se encuentra la solución óptima
        if best_current_gen.fitness.values[0] == NUM_BITS_PROBLEMA:
            print("\n¡Solución óptima encontrada antes de terminar todas las generaciones!")
            break


    # --- 3. Mostrar los Resultados Finales ---
    print("\n--- Evolución Finalizada (DEAP) ---")

    # El mejor individuo de todas las generaciones se encuentra en hof[0]
    best_of_run = hof[0]
    best_candidate_str = "".join(map(str, best_of_run))
    print(f"\nMejor individuo encontrado: {best_candidate_str}")
    # CAMBIO AQUÍ: Formatear como flotante con 0 decimales
    print(f"Aptitud final del mejor individuo: {best_of_run.fitness.values[0]:0.0f}/{NUM_BITS_PROBLEMA}")

    if best_of_run.fitness.values[0] == NUM_BITS_PROBLEMA:
        print("¡El algoritmo genético encontró la solución óptima (todos 1s)!")
    else:
        print("El algoritmo genético no encontró la solución óptima en el número de generaciones dado.")

    # Opcional: Imprimir el logbook completo (estadísticas de todas las generaciones)
    print("\nLogbook completo:")
    print(logbook)

if __name__ == "__main__":
    main()