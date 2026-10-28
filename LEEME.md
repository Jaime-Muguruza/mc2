# Regresión múltiple aplicada a computación científica

Informe y laboratorio interactivo basados en los archivos 9.pdf y Regresión Multiple.pdf del proyecto «Modelos matemáticos de la ciencia 2». Los documentos del curso se citan en el informe; no se redistribuyen dentro de este paquete.

## Archivos principales

- `informe/main.tex`: fuente LaTeX editable, con los 17 puntos solicitados y tres anexos.
- `informe/images/`: cuatro figuras necesarias para compilar.
- `informe/informe.pdf`: informe compilado de 16 páginas.
- `modelo_interactivo.html`: aplicación autónoma para explorar, predecir y simular.
- `datos_y_residuos.csv`: 48 registros, predicciones, residuos o errores, diagnósticos e intervalos.
- `resultados.json`: resultados numéricos y parámetros a precisión completa.
- `reproducir.py` y `requirements.txt`: reproducción del análisis y las figuras.

## Usar el HTML

Haz doble clic en `modelo_interactivo.html` y ábrelo con un navegador que tenga JavaScript habilitado. No requiere internet, instalación ni servidor. No transmite datos.

1. Cambia el trabajo W y el volumen V para obtener un tiempo estimado, un intervalo para la media y un intervalo para una nueva ejecución.
2. Revisa el aviso de dominio: diferencia interpolación dentro de la envolvente convexa, combinaciones sin soporte directo y extrapolación fuera de los rangos.
3. Consulta coeficientes, ANOVA, validación, pruebas y gráficos de residuos.
4. Cambia σ y la semilla en «Genera y ajusta otra muestra». Con σ = 0 se recupera la ecuación generadora. Esta simulación no altera el modelo oficial.
5. Filtra los datos o descarga sus 48 registros. También puedes descargar la muestra interactiva.

El ejemplo inicial W = 60 millones y V = 2,5 GB estima 14,524 s. Su IC del 95% para la media es [14,314; 14,735] s y su IP individual es [13,278; 15,770] s.

## Editar y compilar LaTeX

Mantén `main.tex` junto a `images/`. En Overleaf, carga el contenido de `informe/` como raíz del proyecto y selecciona `main.tex`. Compila con pdfLaTeX dos veces. Localmente, desde la carpeta `informe/`:

```text
pdflatex main.tex
pdflatex main.tex
```

También funciona `tectonic main.tex`. La bibliografía está incluida; no necesita BibTeX. El PDF entregado se compiló con Tectonic y se revisaron las 16 páginas renderizadas.

## Reproducir el análisis

Se utilizó Python 3.12. Desde la carpeta del paquete, se recomienda un entorno virtual:

```text
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python reproducir.py
```

El script escribe `recalculado/`, con CSV, JSON y cuatro figuras en `recalculado/images/`. Conserva los entregables originales. La semilla de los datos es 20261028; la de validación cruzada, 20261029; y la de Monte Carlo, 20261030. El experimento Monte Carlo contiene 5000 ajustes de diseño fijo. La simulación interactiva del HTML usa un generador distinto y no reproduce la secuencia de NumPy.

## Criterios del ejercicio

Los datos son **simulados**; no son mediciones reales ni una base aportada por el docente. Se fijaron 36 filas de ajuste y 12 de prueba antes de estimar, sin reajustar después. Se conserva la notación β₁ para el intercepto de la semana 9. Se aclaran algunas erratas de notación en los materiales. Las observaciones que superan el umbral orientativo de Cook se mantienen y se discute su sensibilidad.

Verificaciones: 48 predicciones e intervalos JavaScript cotejados con Python; identidad ANOVA; simulación determinista y recuperación con σ = 0; entradas inválidas y extrapolación; filtros de registros; sin errores de consola en el navegador de prueba. La revisión del navegador se realizó con un servidor local de comprobación. La aplicación no contiene dependencias de red.
