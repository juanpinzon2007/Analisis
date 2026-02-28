# Proyecto de Metodos Numericos

## Informacion general

El presente proyecto corresponde a una aplicacion de escritorio desarrollada en Python con el proposito de estudiar, visualizar y comparar distintos metodos numericos orientados principalmente a la busqueda de raices de funciones no lineales.

La aplicacion integra una interfaz grafica construida con `tkinter`, graficas generadas con `matplotlib` y modulos propios que implementan los algoritmos numericos utilizados en cada ejercicio.

## Autores

- Juan Andres Pinzon
- Martin Gil
- Alejandro Martinez

## Objetivo

El objetivo del proyecto es ofrecer una herramienta academica que permita:

- aplicar distintos metodos numericos sobre funciones definidas en el curso o practica;
- visualizar el proceso iterativo de cada algoritmo;
- comparar convergencia, error y comportamiento numerico;
- analizar resultados a traves de tablas y representaciones graficas.

## Tecnologias empleadas

- Python
- tkinter
- numpy
- matplotlib
- sympy
- unittest

## Estructura del repositorio

El proyecto se encuentra dentro de carpetas anidadas. La carpeta de trabajo correcta es la que contiene el archivo `main.py`.

```text
Proyecto-metodos-numericos-main/
|-- README.md
`-- Proyecto-metodos-numericos-main/
    |-- .idea/
    `-- Proyecto-metodos-numericos-main/
        `-- proyecto_metodos_numericos/
            |-- main.py
            |-- requirements.txt
            |-- funciones/
            |-- interfaz/
            |-- metodos/
            |-- test/
            `-- utils/
```

Ruta de trabajo:

```powershell
C:\Users\MI PC\Downloads\Proyecto-metodos-numericos-main\Proyecto-metodos-numericos-main\Proyecto-metodos-numericos-main\proyecto_metodos_numericos
```

## Requisitos del sistema

Para ejecutar el proyecto se recomienda contar con:

- Python 3.10 o superior
- `pip`
- soporte para `tkinter`
- sistema operativo Windows con PowerShell o CMD

Version verificada durante la revision:

- Python 3.13.5

## Instalacion

1. Ubicarse en la carpeta raiz descargada:

```powershell
cd "C:\Users\MI PC\Downloads\Proyecto-metodos-numericos-main"
```

2. Ingresar a la carpeta real del proyecto:

```powershell
cd ".\Proyecto-metodos-numericos-main\Proyecto-metodos-numericos-main\proyecto_metodos_numericos"
```

3. Crear un entorno virtual:

```powershell
python -m venv .venv
```

4. Activar el entorno virtual en PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

5. Instalar las dependencias:

```powershell
pip install -r requirements.txt
```

## Dependencias del proyecto

El archivo `requirements.txt` contiene actualmente:

```text
numpy>=1.24
matplotlib>=3.7
sympy>=1.12
```

Revision realizada:

- `requirements.txt` fue revisado frente a los imports reales del proyecto.
- No fue necesario corregirlo.
- `tkinter` no se incluye porque forma parte habitual de la distribucion estandar de Python en Windows.

## Ejecucion de la aplicacion

Una vez instalado el entorno y las dependencias, ejecutar:

```powershell
python main.py
```

Este comando abre la ventana principal del sistema, desde la cual se puede acceder a los diferentes ejercicios numericos mediante un menu lateral.

## Ejecucion de pruebas

Las pruebas disponibles pueden ejecutarse con el siguiente comando:

```powershell
python -m unittest test.test_metodos
```

## Descripcion funcional de la aplicacion

La ventana principal se encuentra definida en `interfaz/main_window.py`. Desde ella se organiza la navegacion entre las distintas vistas del sistema. Cada ejercicio se implementa como una vista independiente compuesta por:

- campos de entrada para parametros numericos;
- tablas de iteraciones;
- graficas de funciones y errores;
- paneles de resultados y analisis.

## Ejercicios implementados en la interfaz

### Ejercicio 1: Metodo de Biseccion

Archivo principal:

- `interfaz/ejercicio1_view.py`

Funcion utilizada:

- `T(lambda) = 2.5 + 0.8 lambda^2 - 3.2 lambda + ln(lambda + 1)`

Metodo implementado:

- `metodos/biseccion.py`

Elementos mostrados:

- intervalo inicial `[a, b]`;
- tolerancia;
- numero maximo de iteraciones;
- tabla del historial iterativo;
- grafica de la funcion;
- grafica del error absoluto en escala logaritmica;
- resultado final con raiz aproximada, iteraciones y tiempo.

### Ejercicio 2: Falsa Posicion vs Biseccion

Archivo principal:

- `interfaz/ejercicio2_view.py`

Funcion utilizada:

- `E(x) = x^3 - 6x^2 + 11x - 6.5`

Metodos implementados:

- `metodos/biseccion.py`
- `metodos/falsa_posicion.py`

Elementos mostrados:

- modo de comparacion o ejecucion individual;
- resumen comparativo;
- tablas de iteraciones para ambos metodos;
- grafica de la funcion;
- grafica de convergencia del error;
- panel final de analisis.

### Ejercicio 3: Metodo de Punto Fijo

Archivo principal:

- `interfaz/ejercicio3_view.py`

Funcion iterativa:

- `g(x) = 0.5 cos(x) + 1.5`

Derivada asociada:

- `g'(x) = -0.5 sin(x)`

Metodo implementado:

- `metodos/punto_fijo.py`

Elementos mostrados:

- seleccion del valor inicial `x0`;
- verificacion de condicion de convergencia;
- tabla de iteraciones;
- cobweb plot;
- comparacion de varios valores iniciales.

### Ejercicio 4: Metodo de Newton-Raphson

Archivo principal:

- `interfaz/ejercicio4_view.py`

Funcion utilizada:

- `T(n) = n^3 - 8n^2 + 20n - 16`

Derivada:

- `T'(n) = 3n^2 - 16n + 20`

Metodo implementado:

- `metodos/newton_raphson.py`

Elementos mostrados:

- ejecucion a partir de un valor inicial;
- comparacion entre distintos valores de `x0`;
- tabla de iteraciones;
- grafica de la funcion;
- representacion de tangentes por iteracion;
- grafica del error absoluto.

### Ejercicio 5: Secante vs Newton-Raphson

Archivo principal:

- `interfaz/ejercicio5_view.py`

Funcion utilizada:

- `P(x) = x e^(-x/2) - 0.3`

Derivada:

- `P'(x) = e^(-x/2) (1 - x/2)`

Elementos mostrados:

- tabla de iteraciones del metodo de la secante;
- tabla de iteraciones de Newton-Raphson;
- tabla comparativa final;
- grafica de la funcion;
- grafica de secantes por iteracion;
- grafica de convergencia del error;
- panel de analisis textual.

## Modulos matematicos incluidos

Dentro de la carpeta `metodos/` se encuentran implementados los siguientes algoritmos:

- `biseccion.py`
- `falsa_posicion.py`
- `punto_fijo.py`
- `newton_raphson.py`
- `secante.py`
- `euler.py`
- `rk4.py`

Los modulos `euler.py` y `rk4.py` forman parte del codigo fuente disponible y pueden utilizarse para futuras ampliaciones del proyecto.

## Funciones matematicas definidas

En `funciones/definiciones.py` se encuentran definidas las funciones utilizadas en los ejercicios:

- `T_lambda`
- `E_workers`
- `g_db`
- `g_db_deriv`
- `T_threads`
- `T_threads_deriv`
- `P_scaling`
- `P_scaling_deriv`

## Archivos principales del sistema

- `main.py`: punto de entrada de la aplicacion
- `requirements.txt`: dependencias externas
- `interfaz/main_window.py`: ventana principal y navegacion
- `funciones/definiciones.py`: definicion de funciones matematicas
- `metodos/`: implementacion de algoritmos numericos
- `utils/validaciones.py`: validaciones auxiliares
- `test/test_metodos.py`: pruebas unitarias disponibles

## Procedimiento recomendado para reproducir el proyecto

1. Descargar o copiar la carpeta completa manteniendo su estructura.
2. Entrar a la carpeta `proyecto_metodos_numericos`.
3. Crear y activar un entorno virtual.
4. Instalar las dependencias indicadas.
5. Ejecutar `python main.py`.
6. Acceder a cada ejercicio desde la interfaz principal.
7. Ejecutar las pruebas con `python -m unittest test.test_metodos` si se desea validar el modulo disponible.

## Comandos utiles

Instalacion de dependencias:

```powershell
pip install -r requirements.txt
```

Ejecucion de la aplicacion:

```powershell
python main.py
```

Ejecucion de pruebas:

```powershell
python -m unittest test.test_metodos
```

## Conclusion

Este proyecto constituye una herramienta academica orientada al analisis de metodos numericos mediante una interfaz grafica clara y modular. Su estructura permite estudiar el comportamiento iterativo de diferentes algoritmos, observar su convergencia y comparar resultados de manera visual y practica. Asimismo, la organizacion del codigo facilita su extension para futuras actividades o ejercicios adicionales.
