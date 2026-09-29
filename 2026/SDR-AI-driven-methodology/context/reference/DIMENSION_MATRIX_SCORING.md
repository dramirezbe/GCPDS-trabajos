# MATRIZ_SCORING.md
## Matriz de scoring para metodologías de desarrollo de software
### Scope: SDR como sensor/herramienta + desarrollo AI-assisted (LLMs, agentes, skills, SDD) con HITL

---

## 1. Propósito

`DIMENSION_ANALYSIS.md` define *qué* dimensiones evaluar y da los extremos (**Débil ↔ Fuerte**)
más un indicador sugerido por cada una. Lo que falta para una matriz **repetible y defendible**
es el **medio** (qué gana un 3 frente a un 1 o un 5) y las **reglas de agregación**. Este archivo
aporta:

1. Una **rúbrica anclada** por dimensión (niveles 1 / 3 / 5, con 2 y 4 como interpolaciones).
2. Puntuación **decimal** (1.0–5.0) para capturar matices y evitar empates artificiales.
3. Reglas para los **dos anclajes** (SDR-sensor ↔ AI-assisted) que son el núcleo del scope.
4. **Agregación con puertas** (composite ponderado + perfil radar + piso Tier 1).

---

## 2. Principios de diseño de la matriz

Una matriz que produce puntajes repetibles y defendibles descansa en cuatro decisiones:

1. **Niveles anclados, no solo extremos.** Se convierte cada *Débil↔Fuerte* en una rúbrica
   conductual (qué comportamiento gana un 1 vs. 3 vs. 5). Sin esto, dos evaluadores puntúan
   distinto la misma metodología — lo que, irónicamente, viola D1.
2. **Puntaje ligado a evidencia.** Cada score cita un artefacto de la metodología (un archivo de
   spec, una traza, un documento de presupuesto de latencia), no una impresión. Si no hay
   artefacto, el score cae por defecto hacia el extremo bajo — esto operacionaliza el criterio
   del marco: *"la ausencia es señal, no ruido"*.
3. **Cobertura de doble ancla.** Cada dimensión tiene *dos* anclas (SDR-sensor y AI-assisted).
   Una metodología dentro del scope debe satisfacer **ambas**, así que se puntúan por separado y
   se combinan (ver §5).
4. **Agregación con puertas.** Un único número ponderado oculta vacíos fatales; se usa junto a un
   **perfil por dimensión** y una **regla de piso Tier 1**.

---

## 3. Escala decimal (1.0 – 5.0)

Puntuación continua con un decimal. Las anclas 1 / 3 / 5 son conductuales; los valores 2 y 4 son
interpolaciones, y los decimales intermedios (p. ej. 3.4) reflejan cumplimiento parcial dentro de
un nivel.

| Rango | Etiqueta | Interpretación |
|-------|----------|----------------|
| 1.0 – 1.9 | **Ausente / Débil** | No aborda la dimensión; ad hoc o inexistente. |
| 2.0 – 2.9 | **Incipiente** | Algo informal en su lugar, más cerca de débil. |
| 3.0 – 3.9 | **Parcial** | Práctica definida pero incompleta o no forzada. |
| 4.0 – 4.9 | **Fuerte parcial** | Fuerte pero con brechas o sin *enforcement* total. |
| 5.0 | **Fuerte / SOTA** | Cumple plenamente el ancla fuerte, con evidencia. |

**Convención de decimales dentro de un nivel:** partir del ancla del nivel (p. ej. 3.0) y sumar
hasta +0.9 según cuántos sub-criterios del *siguiente* nivel ya estén parcialmente presentes.
Ej.: cumple el nivel 3 y además tiene trazas end-to-end (criterio de nivel 5) pero sin evals
continuas → **3.5**.

---

## 4. Rúbrica de puntuación — D1 a D5

Se muestran las anclas **1 / 3 / 5**. Cada fila incluye la evidencia a buscar (derivada del
"indicador sugerido" original). El `min`/`promedio` de doble ancla se resuelve en §5.

### D1 · Reproducibilidad y determinismo
*Pregunta: ¿El mismo insumo produce el mismo resultado, y es auditable/regenerable?*

| Nivel | Ancla conductual | Evidencia a buscar |
|-------|------------------|--------------------|
| **1** | Resultados no repetibles; entradas no capturadas. | — |
| **3** | Entradas capturadas; *o bien* seed *o bien* version pinning; reproducible con esfuerzo manual. | ¿seed **o** versión fijada? |
| **5** | Reproducción bit a bit o estadísticamente acotada; todos los artefactos versionados; regenerado desde la spec, no parcheado. | % de ejecuciones reproducibles; seed/dataset/versión fijados |

### D2 · Puntos de control HITL
*Pregunta: ¿Dónde revisa, aprueba o corrige el humano, y con qué métricas se gobierna ese lazo?*

| Nivel | Ancla conductual | Evidencia a buscar |
|-------|------------------|--------------------|
| **1** | Revisión ad hoc solo al final. | — |
| **3** | Puntos de control definidos pero no basados en riesgo; sin métricas de override/escalación. | nº de gates de aprobación |
| **5** | Puntos de control ubicados por riesgo; tasas de override y escalación medidas; umbrales de confianza calibrados gobiernan el lazo. | nº de gates; tasa de override; tasa de escalación |

### D3 · Fuente de verdad y trazabilidad
*Pregunta: ¿Existe un artefacto autoritativo y versionado del que se genera y contra el que se traza todo (intención → implementación → salida)?*

| Nivel | Ancla conductual | Evidencia a buscar |
|-------|------------------|--------------------|
| **1** | El código/flowgraph es la única verdad; sin trazabilidad. | — |
| **3** | Existe spec versionada pero **no** ejecutable, y la trazabilidad es parcial. | ¿spec versionada? |
| **5** | Spec ejecutable y versionada como única fuente de verdad; traza completa intención→artefacto→salida; regenerable. | ¿spec versionada **y** ejecutable?; % de cobertura de trazabilidad |

### D4 · Evaluación / V&V continua (observabilidad como habilitador)
*Pregunta: ¿La verificación es un evento puntual o un lazo continuo a nivel de sistema, apoyado en trazas observables?*

| Nivel | Ancla conductual | Evidencia a buscar |
|-------|------------------|--------------------|
| **1** | Tests puntuales, sin observabilidad. | — |
| **3** | Tests regulares + trazas parciales; verificación a nivel de componente. | ¿trazas parciales?; cobertura de tests |
| **5** | Evals continuas a nivel de sistema alimentadas por trazas de trayectoria completa; automatizadas (LLM-as-judge/métricas) + revisión humana. | ¿trazas end-to-end?; cobertura de evals; frecuencia de evaluación |

> **Nota:** la rúbrica de D4 ya **absorbe D6 (observabilidad)** — el marco original la lista como
> el *habilitador* de D4 ("sin trazas no hay evaluación"). Por eso "trazas presentes" está
> embebido en las anclas de D4 y reducir la matriz a D1–D5 no pierde la observabilidad.

### D5 · Conciencia de recursos y desempeño
*Pregunta: ¿La metodología modela explícitamente latencia, throughput, cómputo y costo del runtime?*

| Nivel | Ancla conductual | Evidencia a buscar |
|-------|------------------|--------------------|
| **1** | El desempeño se descubre en producción. | — |
| **3** | Presupuestos definidos para *parte* del sistema (p. ej. solo latencia). | ¿presupuesto parcial? |
| **5** | Presupuestos explícitos de latencia/throughput/cómputo/costo, medidos desde el prototipo y forzados. | ¿presupuesto de latencia definido?; ¿costo/operación monitoreado? |

---

## 5. Manejo de los dos anclajes (SDR ↔ AI)

Como la premisa del marco es el *traslape* de dos mundos, se puntúa cada faceta de la dimensión y
luego se combinan. Dos reglas defendibles:

- **Mínimo** — `dim = min(score_SDR, score_AI)`. Alineado con el marco: una metodología unificada
  *"debe cubrir ambos"*. Es severo pero expone metodologías que solo cubren un lado.
- **Promedio con bandera de desbalance** — más suave; se reporta la brecha cuando
  `|score_SDR − score_AI| ≥ 2`, para que una metodología desbalanceada no se esconda tras un
  promedio medio.

**Recomendación:** usar **`min`** en **D1, D2 y D8** (la §6 del marco las nombra como las de mayor
tensión al cruzar ambos mundos) y **promedio-con-bandera** en el resto.

| Dimensión | Regla de combinación |
|-----------|----------------------|
| D1 | `min(SDR, AI)` |
| D2 | `min(SDR, AI)` |
| D3 | promedio + bandera si \|Δ\| ≥ 2 |
| D4 | promedio + bandera si \|Δ\| ≥ 2 |
| D5 | promedio + bandera si \|Δ\| ≥ 2 |

---

## 6. Agregación, pesos y composite

Pesos por tier del marco original: **Tier 1 ×3, Tier 2 ×2, Tier 3 ×1**.
Para D1–D5: D1–D4 son Tier 1; **D5 es Tier 2**.

- Máximo composite (solo D1–D5) = (4 dims × 5 × 3) + (1 dim × 5 × 2) = 60 + 10 = **70**.
- Normalizar a 0–100: `composite / 70 × 100`, para mantener comparabilidad aunque se use un
  subconjunto distinto de dimensiones.

**No liderar con el composite.** Acompañarlo siempre de:

- **Perfil radar** por metodología (sugerido en el marco) — hace visibles las brechas.
- **Regla de piso Tier 1:** si *cualquier* dimensión Tier 1 (D1–D4) puntúa **< 2.0**, la
  metodología queda **descalificada** sin importar el total. Una metodología fuerte en todo pero
  con trazabilidad nula (D3≈1) está estructuralmente rota, y un composite alto lo enmascararía.

### Fórmula de ejemplo

```
composite   = 3·(D1 + D2 + D3 + D4) + 2·D5
normalizado = composite / 70 × 100
descalifica = (min(D1, D2, D3, D4) < 2.0)
```

---

## 7. Plantilla de matriz (D1–D5)

| Dimensión | Tier | Peso | Score SDR | Score AI | Regla | **Score final** | Evidencia (artefacto citado) |
|-----------|------|------|-----------|----------|-------|-----------------|------------------------------|
| D1 · Reproducibilidad/determinismo | 1 | ×3 | _ | _ | min | _ | |
| D2 · Puntos de control HITL | 1 | ×3 | _ | _ | min | _ | |
| D3 · Fuente de verdad/trazabilidad | 1 | ×3 | _ | _ | prom | _ | |
| D4 · Evaluación/V&V continua | 1 | ×3 | _ | _ | prom | _ | |
| D5 · Recursos/desempeño | 2 | ×2 | _ | _ | prom | _ | |
| **Composite (máx 70)** | | | | | | **_** | |
| **Normalizado (0–100)** | | | | | | **_** | |
| **¿Descalifica? (Tier 1 < 2.0)** | | | | | | **_** | |

*(Extender columnas a la derecha para Metodología A, B, C… o duplicar la tabla por metodología.)*

---

## 8. Cómo hacer reproducible el propio scoring

Conviene, porque el marco lo exige: el proceso de puntuación debe satisfacer **D1** y **D2**.
Concretamente — **dos evaluadores independientes** puntúan desde la rúbrica anclada, se calcula el
**acuerdo** y se **reconcilian** los desacuerdos contra la evidencia citada (ese paso de
reconciliación *es* un punto de control humano, D2). Así la matriz no se convierte en opinión
disfrazada de números.

---

## 9. Extensión al set completo (D6–D13)

Este archivo desarrolla la rúbrica para D1–D5. Para las demás, aplicar el mismo patrón (anclas
1/3/5 + evidencia + regla de doble ancla), recordando las tensiones del marco:

- **D8** (fidelidad test↔prod): usar **`min`**, es una de las tres dimensiones de mayor tensión.
- **D12** (gobernanza): poco desarrollada en SDR-sensor; su ausencia es *señal* — puntuar bajo,
  no omitir.
- **D5 vs. metodologías AI:** recursos suele ignorarse en AI-assisted; una metodología unificada
  fuerte debe cubrir ambos lados.

---

## 10. Referencia a las dimensiones originales (sin pérdida)

Se conserva el catálogo completo del marco (`DIMENSION_ANALYSIS.md` §4) como referencia rápida.

### Tier 1 — Núcleo

| ID | Dimensión | Pregunta de evaluación |
|----|-----------|------------------------|
| D3 ★ | Fuente de verdad y trazabilidad | ¿Artefacto autoritativo y versionado, trazable intención→salida? |
| D1 ★ | Reproducibilidad y determinismo | ¿Mismo insumo → mismo resultado, auditable/regenerable? |
| D4 ★ | Evaluación/V&V continua | ¿Verificación puntual o lazo continuo a nivel de sistema? |
| D2 ★ | Puntos de control HITL | ¿Dónde revisa/aprueba/corrige el humano y con qué métricas? |
| D6 | Observabilidad de ejecución | ¿Se puede ver la trayectoria completa de lo que hizo el sistema? |

### Tier 2 — Estructurales

| ID | Dimensión | Pregunta de evaluación |
|----|-----------|------------------------|
| D7 | Modularidad, composabilidad y ecosistema | ¿Se ensambla de componentes probados por separado?, ¿hay registro reutilizable? |
| D5 ★ | Conciencia de recursos y desempeño | ¿Modela latencia, throughput, cómputo y costo del runtime? |
| D8 | Fidelidad test↔producción + record/replay | ¿Qué tan fiel es el test a producción?, ¿puedes reproducir escenarios reales? |
| D9 | Procedencia y gestión de datos/contexto | ¿Cómo se captura, versiona y fundamenta el insumo? |
| D10 | Automatización del pipeline (CI/CD + generación) | ¿Qué fracción de build→test→deploy está automatizada? |

### Tier 3 — Contextuales

| ID | Dimensión | Pregunta de evaluación |
|----|-----------|------------------------|
| D11 | Reconfiguración runtime / evolución post-despliegue | ¿Cambia comportamiento sin redeploy completo? |
| D12 | Gobernanza, seguridad y cumplimiento | ¿Guardrails, control de acceso, cumplimiento regulatorio? |
| D13 | Costo/esfuerzo total y velocidad de iteración | ¿Reduce costo (dev+operación) y latencia idea→prototipo verificable? |

---

*Documento derivado de `DIMENSION_ANALYSIS.md`. La rúbrica de scoring está diseñada para producir
puntajes decimales (1.0–5.0) repetibles, ligados a evidencia y comparables entre metodologías
dentro del scope indicado.*