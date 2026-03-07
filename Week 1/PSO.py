# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 01:25:59 2025

@author: MARCELOFGB
"""

import numpy as np
import matplotlib.pyplot as plt
import random # Usaremos random para inicialización, numpy.random para lo demás

# --- 1. Definición de la Función Objetivo (Problema de los Drones) ---

def funcion_objetivo_drones(posicion):
    """
    Función de costo para una formación de drones.
    Minimizar esta función significa encontrar la mejor formación.

    La 'posicion' de la partícula representa los parámetros de la formación:
    posicion[0]: ancho de la formación (ej. en metros)
    posicion[1]: alto de la formación (ej. en metros)

    FACTORES CLIMÁTICOS Y CONDICIONES (fijos para este ejemplo):
    Viento_velocidad: Mayor viento -> más penalización por área expuesta.
    Turbulencia_intensidad: Mayor turbulencia -> más penalización por formas desequilibradas.
    Densidad_penalizacion: Cuánto penalizar por tener los drones muy juntos.
    """
    ancho_formacion = posicion[0]
    alto_formacion = posicion[1]
    
    # Asegurar que los parámetros sean positivos y dentro de límites razonables
    # (Penalización si se salen de rango)
    if ancho_formacion <= 0.1 or alto_formacion <= 0.1:
        return 1e10 # Costo muy alto si los valores son muy pequeños o negativos
    
    if ancho_formacion > 100 or alto_formacion > 100:
        return 1e10 # Costo muy alto si los valores son muy grandes

    # Condiciones específicas de clima y viento (pueden ser dinámicas en problemas reales)
    VELOCIDAD_VIENTO = 15.0 # m/s
    INTENSIDAD_TURBULENCIA = 0.8 # De 0 a 1, qué tan "movido" está el aire
    NUM_DRONES = 25
    MIN_DISTANCIA_DRONES = 2.0 # Distancia mínima deseada entre drones (en metros)

    # --- Componentes del costo ---

    # 1. Costo por Resistencia al Viento (Mayor área frontal = mayor costo)
    costo_viento = VELOCIDAD_VIENTO * (ancho_formacion + alto_formacion) # Suma de dimensiones para una estimación simple de área expuesta

    # 2. Costo por Riesgo de Colisión / Densidad (Menor área total para 25 drones = mayor riesgo)
    # Calculamos el área que ocupa la formación
    area_formacion = ancho_formacion * alto_formacion
    
    # Asumimos que los 25 drones necesitan un espacio mínimo.
    # Si el área es muy pequeña, el costo aumenta drásticamente.
    # Un área_minima_segura_estimada = NUM_DRONES * (MIN_DISTANCIA_DRONES ** 2)
    # Por ejemplo, si cada drone necesita ~4m^2, 25 drones necesitan 100m^2
    area_minima_ideal_por_drone = MIN_DISTANCIA_DRONES**2
    area_minima_total_ideal = NUM_DRONES * area_minima_ideal_por_drone 
    
    # Penalizamos si el área es menor que la ideal
    # Usamos una penalización inversa: cuanto menor el área, mayor el costo
    costo_densidad = (area_minima_total_ideal / (area_formacion + 0.001)) * 50 # El +0.001 evita división por cero

    # 3. Costo por Estabilidad / Maniobrabilidad (Formas desequilibradas vs. formas más "cuadradas")
    # Penalizamos si la relación ancho/alto es muy extrema (ej. un hilo largo o una torre alta)
    relacion_aspecto = max(ancho_formacion, alto_formacion) / min(ancho_formacion, alto_formacion + 0.001)
    costo_estabilidad = INTENSIDAD_TURBULENCIA * (relacion_aspecto - 1.0) * 10 # Penaliza desviación de una relación 1:1

    # Costo Total
    costo_total = costo_viento + costo_densidad + costo_estabilidad
    
    return costo_total

# --- 2. Implementación de una Partícula ---

class Particula:
    def __init__(self, dimensiones, limites_posicion):
        """
        Inicializa una partícula.
        dimensiones: número de parámetros a optimizar (ej. 2 para ancho y alto de formación)
        limites_posicion: tupla de (min_valor, max_valor) para cada dimensión
        """
        self.dimensiones = dimensiones
        self.limites_posicion = limites_posicion

        # Posición inicial aleatoria dentro de los límites
        self.posicion = np.array([random.uniform(lim[0], lim[1]) for lim in limites_posicion])
        
        # Velocidad inicial aleatoria
        self.velocidad = np.array([random.uniform(-1, 1) for _ in range(dimensiones)])
        
        # Mejor posición personal y su valor (se inicializa con un valor muy alto para minimizar)
        self.pbest_posicion = np.copy(self.posicion)
        self.pbest_valor = float('inf') # Infinito positivo

    def actualizar_pbest(self, valor_actual_funcion_objetivo):
        """Actualiza la mejor posición personal si la actual es mejor."""
        if valor_actual_funcion_objetivo < self.pbest_valor:
            self.pbest_valor = valor_actual_funcion_objetivo
            self.pbest_posicion = np.copy(self.posicion)

    def actualizar_velocidad(self, gbest_posicion, w, c1, c2):
        """
        Actualiza la velocidad de la partícula.
        w: peso de inercia
        c1: coeficiente cognitivo (atracción a pbest)
        c2: coeficiente social (atracción a gbest)
        """
        # Componente de inercia
        inercia = w * self.velocidad

        # Componente cognitivo (atracción a la mejor posición personal)
        r1 = np.random.rand(self.dimensiones)
        cognitivo = c1 * r1 * (self.pbest_posicion - self.posicion)

        # Componente social (atracción a la mejor posición global del enjambre)
        r2 = np.random.rand(self.dimensiones)
        social = c2 * r2 * (gbest_posicion - self.posicion)

        self.velocidad = inercia + cognitivo + social

        # Opcional: Limitar la velocidad para evitar que las partículas se salgan de control
        v_max = (self.limites_posicion[0][1] - self.limites_posicion[0][0]) * 0.1 # 10% del rango de la primera dimensión
        self.velocidad = np.clip(self.velocidad, -v_max, v_max)


    def actualizar_posicion(self):
        """Actualiza la posición de la partícula y la mantiene dentro de los límites."""
        self.posicion += self.velocidad
        
        # Asegurar que la posición esté dentro de los límites definidos
        for i in range(self.dimensiones):
            self.posicion[i] = np.clip(self.posicion[i], self.limites_posicion[i][0], self.limites_posicion[i][1])

# --- 3. Algoritmo PSO ---

def pso_optimizar(funcion_objetivo, dimensiones, limites_posicion, num_particulas=30, max_iteraciones=100,
                  w=0.5, c1=1.5, c2=1.5):
    """
    Ejecuta el algoritmo PSO.
    
    funcion_objetivo: La función a minimizar.
    dimensiones: Número de dimensiones del problema.
    limites_posicion: Lista de tuplas (min, max) para cada dimensión.
    num_particulas: Número de partículas en el enjambre.
    max_iteraciones: Número máximo de iteraciones.
    w, c1, c2: Parámetros del PSO.
    """
    # Inicialización del enjambre
    enjambre = [Particula(dimensiones, limites_posicion) for _ in range(num_particulas)]

    # Mejor posición global encontrada por el enjambre y su valor
    gbest_posicion = np.array([random.uniform(lim[0], lim[1]) for lim in limites_posicion]) # Inicializa un gbest arbitrario
    gbest_valor = float('inf')

    # Historial para seguimiento (didáctico)
    historial_gbest_valor = []

    print("\n--- INICIO OPTIMIZACIÓN PSO ---")
    for iteracion in range(max_iteraciones):
        for particula in enjambre:
            # Evaluar la función objetivo para la posición actual de la partícula
            valor_actual = funcion_objetivo(particula.posicion)

            # Actualizar la mejor posición personal de la partícula
            particula.actualizar_pbest(valor_actual)

            # Actualizar la mejor posición global del enjambre
            if valor_actual < gbest_valor:
                gbest_valor = valor_actual
                gbest_posicion = np.copy(particula.posicion)

        # Actualizar velocidades y posiciones de todas las partículas
        for particula in enjambre:
            particula.actualizar_velocidad(gbest_posicion, w, c1, c2)
            particula.actualizar_posicion()
        
        historial_gbest_valor.append(gbest_valor)

        # Mostrar progreso (opcional)
        if (iteracion + 1) % 10 == 0 or iteracion == 0:
            print(f"Iteración {iteracion + 1}/{max_iteraciones} - Mejor Costo Global: {gbest_valor:.4f} "
                  f"- Formación: Ancho={gbest_posicion[0]:.2f}m, Alto={gbest_posicion[1]:.2f}m")

    print("\n--- OPTIMIZACIÓN PSO FINALIZADA ---")
    print(f"Mejor Formación Encontrada (Dimensiones): Ancho={gbest_posicion[0]:.2f}m, Alto={gbest_posicion[1]:.2f}m")
    print(f"Mejor Costo (riesgo/ineficiencia): {gbest_valor:.4f}")

    return gbest_posicion, gbest_valor, historial_gbest_valor


# --- 4. Función de Visualización de Formación de Drones ---

def dibujar_formacion(ancho, alto, titulo="", num_drones=25):
    """
    Dibuja una formación rectangular de drones y anota sus dimensiones y costo.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Calcular posiciones de los drones en una cuadrícula
    # Para 25 drones, una cuadrícula de 5x5 es ideal
    sqrt_drones = int(np.sqrt(num_drones))
    if sqrt_drones * sqrt_drones != num_drones:
        print(f"Advertencia: No se pueden distribuir {num_drones} drones en una cuadrícula perfecta.")
        # Fallback a una distribución más simple o generar error.
        # Por simplicidad, si no es cuadrado perfecto, ajustaremos a un grid lo más cercano posible
        rows = int(np.ceil(np.sqrt(num_drones)))
        cols = int(np.ceil(num_drones / rows))
    else:
        rows = cols = sqrt_drones

    x_drone = np.linspace(0, ancho, cols)
    y_drone = np.linspace(0, alto, rows)
    
    # Generar todos los puntos de la cuadrícula
    drone_x_coords, drone_y_coords = np.meshgrid(x_drone, y_drone)
    
    # Aplanar los arrays para obtener una lista de coordenadas (x,y)
    drone_x_coords = drone_x_coords.flatten()
    drone_y_coords = drone_y_coords.flatten()

    ax.scatter(drone_x_coords[:num_drones], drone_y_coords[:num_drones], color='blue', s=100, marker='^', label='Dron')
    
    # Dibujar el contorno de la formación
    ax.plot([0, ancho, ancho, 0, 0], [0, 0, alto, alto, 0], 'r--', label='Contorno Formación')

    ax.set_title(titulo)
    ax.set_xlabel(f"Ancho (m): {ancho:.2f}")
    ax.set_ylabel(f"Alto (m): {alto:.2f}")
    ax.set_xlim(-ancho * 0.1, ancho * 1.1)
    ax.set_ylim(-alto * 0.1, alto * 1.1)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    plt.show()

# --- 5. Ejecución del Ejemplo Completo ---

if __name__ == "__main__":
    print("DEMOSTRACIÓN DIDÁCTICA DE PSO PARA OPTIMIZAR FORMACIÓN DE DRONES")

    # Parámetros del problema de optimización
    num_dimensiones = 2 # Ancho y Alto
    # Límites para el ancho y el alto de la formación (ej. entre 5 y 50 metros)
    limites = [(5.0, 50.0), (5.0, 50.0)] 

    # Parámetros del algoritmo PSO
    num_particulas = 50
    num_iteraciones = 100
    inercia = 0.7  # w
    atraccion_cognitiva = 1.8 # c1
    atraccion_social = 1.8   # c2

    # Ejecutar el PSO
    mejor_formacion_parametros, mejor_costo, historial = pso_optimizar(
        funcion_objetivo_drones, num_dimensiones, limites,
        num_particulas, num_iteraciones,
        inercia, atraccion_cognitiva, atraccion_social
    )

    print("\n--- RESULTADOS ---")
    print("Parámetros de la mejor formación encontrada por PSO:")
    print(f"  Ancho: {mejor_formacion_parametros[0]:.2f} metros")
    print(f"  Alto:  {mejor_formacion_parametros[1]:.2f} metros")
    print(f"  Costo asociado: {mejor_costo:.4f}")

    # --- Visualización de la convergencia del PSO ---
    plt.figure(figsize=(10, 5))
    plt.plot(historial, color='purple')
    plt.title('Convergencia del Algoritmo PSO: Mejor Costo Global por Iteración')
    plt.xlabel('Iteración')
    plt.ylabel('Mejor Costo de Formación')
    plt.grid(True)
    plt.show()

    # --- Dibujar la mejor formación encontrada por el PSO ---
    dibujar_formacion(mejor_formacion_parametros[0], mejor_formacion_parametros[1],
                      titulo=f"Mejor Formación Encontrada por PSO\nCosto: {mejor_costo:.2f}")

    # --- Comparación con Formaciones "Factibles" Predefinidas (Al menos 4) ---
    print("\n--- COMPARACIÓN CON FORMACIONES FACTIBLES PREDEFINIDAS ---")

    formaciones_factibles = {
        "Formación Cuadrada Estándar (25m x 25m)": (25.0, 25.0),
        "Formación Ancha y Baja (40m x 10m)": (40.0, 10.0),
        "Formación Alta y Estrecha (10m x 40m)": (10.0, 40.0),
        "Formación Compacta (15m x 15m)": (15.0, 15.0),
    }

    print("Evaluando formaciones predefinidas (no optimizadas por PSO):")
    for nombre, (ancho_f, alto_f) in formaciones_factibles.items():
        costo_f = funcion_objetivo_drones(np.array([ancho_f, alto_f]))
        print(f"- {nombre}: Ancho={ancho_f:.1f}m, Alto={alto_f:.1f}m, Costo = {costo_f:.2f}")
        dibujar_formacion(ancho_f, alto_f, titulo=f"{nombre}\nCosto: {costo_f:.2f}")

    # Reflexión final
    print("\n--- REFLEXIÓN ---")
    print("Como puedes observar, el PSO busca en el espacio de parámetros (ancho, alto) para encontrar")
    print("la combinación que minimiza el 'costo' de la formación, según los criterios de viento, densidad y estabilidad.")
    print("Normalmente, la solución de PSO será mejor (menor costo) que la mayoría de las formaciones predefinidas simples,")
    print("demostrando su capacidad para explorar el espacio de búsqueda eficientemente.")