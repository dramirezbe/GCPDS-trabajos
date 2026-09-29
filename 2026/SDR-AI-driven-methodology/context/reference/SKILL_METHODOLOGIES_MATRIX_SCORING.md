---
name: methodologies-matrix-scoring
description: >
  Genera una matriz de scoring comparativa para metodologías de desarrollo de
  software, dentro del scope "SDR como sensor/herramienta + desarrollo
  AI-assisted (LLMs, agentes, skills, SDD) con HITL". Recibe como entrada una
  lista de metodologías explicadas (pegadas en el prompt o adjuntas) y produce,
  para cada una, un score decimal (1.0–5.0) por dimensión usando D1–D5
  (opcionalmente D6–D13), aplicando la regla de doble ancla (SDR ↔ AI), los
  pesos por tier, el composite normalizado y la puerta de piso Tier 1. Úsala
  cuando pidan "comparar metodologías", "matriz de scoring", "puntuar/rankear
  metodologías" o "evaluar contra las dimensiones D1–D5".
---

# Skill · Matriz de scoring para metodologías de desarrollo de software

## 0. Qué hace esta skill

Convierte una **lista de metodologías explicadas** en una **matriz de scoring
comparable, repetible y defendible**. Para cada metodología asigna un puntaje
decimal (1.0–5.0) por dimensión usando la rúbrica anclada de abajo, combina las
dos facetas del scope (SDR-sensor ↔ AI-assisted), agrega con pesos por tier,
normaliza a 0–100 y aplica una puerta de descalificación.

**Diseñada para LLMs no agénticos.** No abre ni busca archivos. Toda la rúbrica
está embebida aquí. La entrada (las metodologías) llega **pegada en el prompt o
adjunta**; si falta, pídela antes de puntuar.

**Scope fijo:** SDR como sensor/herramienta **+** desarrollo AI-assisted (LLMs,
agentes, skills, SDD) con humano en el lazo (HITL). Cada dimensión se evalúa
contra *ambos* mundos (ver §4).

---

## 1. Entrada esperada

- Una **lista de metodologías**, cada una con una explicación en prosa
  (descripción, prácticas, artefactos, flujo de trabajo, herramientas).
- Opcionalmente, el **subconjunto de dimensiones** a usar. Por defecto: **D1–D5**.
  Si piden "todas" o mencionan D6–D13, incluye las del catálogo §8.
- Opcionalmente, la **regla de combinación** preferida (min / promedio). Por
  defecto usa la asignación de §5.

### 1.1 Criterios de categorización conceptual

Cuando el conjunto de metodologías suministrado contenga enfoques heterogéneos, pueden agruparse o etiquetarse según su foco primario de diseño:

- **Enfoque IA-assisted / Agéntico (Agnóstico de dominio):**
  Metodologías centradas en el ciclo agéntico, especificaciones formales/ejecutables, flujos de desarrollo asistido por LLMs y evaluación continua. Suelen exhibir alta madurez en el lazo de IA y puntos de control humano, pero carecen de modelado nativo de señales RF, hardware físico o captura de datos IQ.
- **Enfoque de Dominio / SDR-sensor (Software y estándares de ingeniería):**
  Estándares, frameworks y prácticas de ingeniería de software para SDR/DSP. Poseen alto rigor en adquisición de datos IQ, metadatos, calibración física, repetibilidad determinista y presupuestos de tiempo real, pero carecen de lazo asistido por IA o generación agéntica.
- **Enfoque Híbrido / Intersección (AI + SDR):**
  Propuestas o herramientas que integran utilidades y APIs de dominio SDR con modelos de lenguaje, agentes o skills preempaquetadas.
- **Enfoque Normativo / Gobernanza:**
  Estándares y marcos regulatorios de gobernanza de IA y supervisión humana que establecen requisitos formales de control (HITL/HOTL) según el nivel de criticidad o riesgo del sistema.

Si la entrada no trae suficiente detalle para anclar un score con evidencia,
**no inventes**: puntúa hacia el extremo bajo y anótalo (principio *"la ausencia
es señal, no ruido"*), o solicita el artefacto faltante.

---

## 2. Principios de puntuación (no negociables)

1. **Niveles anclados, no solo extremos.** Puntúa contra la conducta descrita en
   la rúbrica (§4), no contra una impresión general.
2. **Score ligado a evidencia.** Cada puntaje debe citar un elemento concreto de
   la explicación de la metodología (una spec, una traza, un presupuesto de
   latencia, un gate). Sin artefacto → el score cae al extremo bajo.
3. **Cobertura de doble ancla.** Cada dimensión tiene dos facetas (SDR-sensor y
   AI-assisted). Puntúa cada una por separado y combínalas (§5).
4. **Agregación con puertas.** Nunca lideres con el número único: acompáñalo del
   perfil por dimensión y de la puerta de piso Tier 1 (§6).

---

## 3. Escala decimal (1.0 – 5.0)

Puntuación continua con un decimal. Las anclas 1/3/5 son conductuales; 2 y 4 son
interpolaciones; los decimales reflejan cumplimiento parcial dentro de un nivel.

| Rango | Etiqueta | Interpretación |
|-------|----------|----------------|
| 1.0 – 1.9 | **Ausente / Débil** | No aborda la dimensión; ad hoc o inexistente. |
| 2.0 – 2.9 | **Incipiente** | Algo informal en su lugar, más cerca de débil. |
| 3.0 – 3.9 | **Parcial** | Práctica definida pero incompleta o no forzada. |
| 4.0 – 4.9 | **Fuerte parcial** | Fuerte pero con brechas o sin *enforcement* total. |
| 5.0 | **Fuerte / SOTA** | Cumple plenamente el ancla fuerte, con evidencia. |

**Decimales dentro de un nivel:** parte del ancla del nivel (p. ej. 3.0) y suma
hasta +0.9 según cuántos sub-criterios del *siguiente* nivel ya estén
parcialmente presentes. Ej.: cumple nivel 3 y además tiene trazas end-to-end
(criterio nivel 5) pero sin evals continuas → **3.5**.

---

## 4. Rúbrica de puntuación — D1 a D5

Anclas **1 / 3 / 5** con la evidencia a buscar. La combinación de doble ancla se
resuelve en §5.

### D1 · Reproducibilidad y determinismo
*¿El mismo insumo produce el mismo resultado, y es auditable/regenerable?*

| Nivel | Ancla conductual | Evidencia a buscar |
|-------|------------------|--------------------|
| **1** | Resultados no repetibles; entradas no capturadas. | — |
| **3** | Entradas capturadas; *o bien* seed *o bien* version pinning; reproducible con esfuerzo manual. | ¿seed **o** versión fijada? |
| **5** | Reproducción bit a bit o estadísticamente acotada; todos los artefactos versionados; regenerado desde la spec, no parcheado. | % de ejecuciones reproducibles; seed/dataset/versión fijados |

### D2 · Puntos de control HITL
*¿Dónde revisa/aprueba/corrige el humano, y con qué métricas se gobierna el lazo?*

| Nivel | Ancla conductual | Evidencia a buscar |
|-------|------------------|--------------------|
| **1** | Revisión ad hoc solo al final. | — |
| **3** | Puntos de control definidos pero no basados en riesgo; sin métricas de override/escalación. | nº de gates de aprobación |
| **5** | Puntos de control ubicados por riesgo; tasas de override y escalación medidas; umbrales de confianza calibrados gobiernan el lazo. | nº de gates; tasa de override; tasa de escalación |

### D3 · Fuente de verdad y trazabilidad
*¿Existe un artefacto autoritativo y versionado del que se genera y contra el que se traza todo (intención → implementación → salida)?*

| Nivel | Ancla conductual | Evidencia a buscar |
|-------|------------------|--------------------|
| **1** | El código/flowgraph es la única verdad; sin trazabilidad. | — |
| **3** | Existe spec versionada pero **no** ejecutable, y la trazabilidad es parcial. | ¿spec versionada? |
| **5** | Spec ejecutable y versionada como única fuente de verdad; traza completa intención→artefacto→salida; regenerable. | ¿spec versionada **y** ejecutable?; % de cobertura de trazabilidad |

### D4 · Evaluación / V&V continua (observabilidad como habilitador)
*¿La verificación es un evento puntual o un lazo continuo a nivel de sistema, apoyado en trazas observables?*

| Nivel | Ancla conductual | Evidencia a buscar |
|-------|------------------|--------------------|
| **1** | Tests puntuales, sin observabilidad. | — |
| **3** | Tests regulares + trazas parciales; verificación a nivel de componente. | ¿trazas parciales?; cobertura de tests |
| **5** | Evals continuas a nivel de sistema alimentadas por trazas de trayectoria completa; automatizadas (LLM-as-judge/métricas) + revisión humana. | ¿trazas end-to-end?; cobertura de evals; frecuencia de evaluación |

> **Nota:** D4 ya **absorbe D6 (observabilidad)** — el marco la lista como el
> *habilitador* de D4 ("sin trazas no hay evaluación"). Por eso "trazas
> presentes" está embebido en las anclas de D4, y reducir la matriz a D1–D5 no
> pierde la observabilidad.

### D5 · Conciencia de recursos y desempeño
*¿La metodología modela explícitamente latencia, throughput, cómputo y costo del runtime?*

| Nivel | Ancla conductual | Evidencia a buscar |
|-------|------------------|--------------------|
| **1** | El desempeño se descubre en producción. | — |
| **3** | Presupuestos definidos para *parte* del sistema (p. ej. solo latencia). | ¿presupuesto parcial? |
| **5** | Presupuestos explícitos de latencia/throughput/cómputo/costo, medidos desde el prototipo y forzados. | ¿presupuesto de latencia definido?; ¿costo/operación monitoreado? |

---

## 5. Combinación de los dos anclajes (SDR ↔ AI)

Puntúa cada dimensión **dos veces** — faceta SDR-sensor y faceta AI-assisted — y
combínalas:

- **Mínimo** — `dim = min(score_SDR, score_AI)`. Severo; expone metodologías que
  solo cubren un lado. Se usa donde el marco marca mayor tensión.
- **Promedio con bandera de desbalance** — `dim = (score_SDR + score_AI) / 2`, y
  se **reporta la brecha** cuando `|score_SDR − score_AI| ≥ 2`, para que una
  metodología desbalanceada no se esconda tras un promedio medio.

**Asignación por defecto** (D1, D2 son de mayor tensión → `min`):

| Dimensión | Regla de combinación |
|-----------|----------------------|
| D1 | `min(SDR, AI)` |
| D2 | `min(SDR, AI)` |
| D3 | promedio + bandera si \|Δ\| ≥ 2 |
| D4 | promedio + bandera si \|Δ\| ≥ 2 |
| D5 | promedio + bandera si \|Δ\| ≥ 2 |

Si se extiende a D6–D13, usa `min` también en **D8** (fidelidad test↔prod).

---

## 6. Agregación, pesos, composite y puertas

**Pesos por tier:** Tier 1 ×3, Tier 2 ×2, Tier 3 ×1. Para D1–D5: **D1–D4 son
Tier 1 (×3)**, **D5 es Tier 2 (×2)**.

```
composite   = 3·(D1 + D2 + D3 + D4) + 2·D5
normalizado = composite / 70 × 100          # 70 = máximo con solo D1–D5
descalifica = ( min(D1, D2, D3, D4) < 2.0 ) # puerta de piso Tier 1
```

- Máximo composite (solo D1–D5) = (4 × 5 × 3) + (1 × 5 × 2) = 60 + 10 = **70**.
- Normalizar a 0–100 mantiene comparabilidad aunque se use otro subconjunto de
  dimensiones (recalcula el máximo si cambias el set).
- **No lideres con el composite.** Acompáñalo siempre de:
  - **Perfil radar / por dimensión** de cada metodología (hace visibles las brechas).
  - **Puerta de piso Tier 1:** si *cualquier* dimensión Tier 1 (D1–D4) < **2.0**,
    la metodología queda **DESCALIFICADA** sin importar el total. (Una
    metodología fuerte en todo pero con trazabilidad nula, D3≈1, está
    estructuralmente rota; el composite alto lo enmascararía.)

Si extiendes el set, recalcula el máximo: `Σ(peso_dim × 5)`.

---

## 7. Formato de salida

Produce la evaluación en una estructura consolidada, directa y de alta legibilidad:

### 7.1 Matriz consolidada de scoring (única matriz)

Una **única tabla principal** con todas las metodologías evaluadas en las filas y las dimensiones evaluadas en las columnas:

| Categoría / Tipo | ID | Metodología | D1 (×3) | D2 (×3) | D3 (×3) | D4 (×3) | D5 (×2) | Composite (/70) | Normalizado (/100) | Estado (Puerta Tier 1) |
|:---:|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| IA-assisted | M1 | [Nombre Metodología 1] | 1.5 | 2.0 | 3.0 🚩 | 3.0 🚩 | 1.5 | 31.5 | 45.0 | ❌ Descalificada (D1 < 2.0) |
| SDR-sensor | M2 | [Nombre Metodología 2] | 4.0 | 1.5 | 3.0 | 2.0 | 2.0 | 35.5 | 50.7 | ❌ Descalificada (D2 < 2.0) |
| Híbrida | M3 | [Nombre Metodología 3] | 3.5 | 3.0 | 3.0 | 3.0 | 2.0 | 39.5 | 56.4 | ✅ Calificada |

- **Valores en celdas (D1–D5):** Score final combinado según la regla de §5.
- **Banderas (🚩):** Marca con 🚩 cualquier dimensión con desbalance severo de doble ancla (`|SDR − AI| ≥ 2`).
- **Estado:** Indica **✅ Calificada** o **❌ Descalificada (Dim < 2.0)** aplicando la puerta de piso Tier 1 (§6: si `min(D1..D4) < 2.0`).

### 7.2 Hallazgos estructurales y síntesis

Presenta un análisis conciso derivado de la matriz:
1. **Mecanismo de descalificación:** Cuántas y cuáles metodologías caen por la puerta Tier 1 y los patrones de fallo observados (p. ej., desbalance por falta de modelado SDR o ausencia de lazo agéntico AI).
2. **Brechas de doble ancla:** Dimensiones con banderas 🚩 recurrentes y techos o huecos sistémicos comunes.
3. **Implicación composicional:** Identificar qué metodologías aportan las facetas más sólidas por dimensión para construir una metodología compuesta viable.

### 7.3 Desglose de evidencia y facetas (Notas compactas)

En lugar de matrices adicionales por metodología, documenta los puntajes individuales (SDR / AI), la regla aplicada y la justificación en una lista concisa:

- **[ID] Nombre de la metodología:**
  - **D1:** SDR = X.X, AI = Y.Y (`min`) → **Final**. *Evidencia:* [artefacto citado o justificación del score bajo].
  - **D2:** SDR = X.X, AI = Y.Y (`min`) → **Final**. *Evidencia:* ...
  - **D3:** SDR = X.X, AI = Y.Y (`prom ΔZ.Z 🚩`) → **Final**. *Evidencia:* ...
  - **D4:** SDR = X.X, AI = Y.Y (`prom`) → **Final**. *Evidencia:* ...
  - **D5:** SDR = X.X, AI = Y.Y (`prom`) → **Final**. *Evidencia:* ...

---

## 8. Extensión opcional al set completo (D6–D13)

Aplica el mismo patrón (anclas 1/3/5 + evidencia + regla de doble ancla).

**Tier 1 — Núcleo (×3):** D3 ★, D1 ★, D4 ★, D2 ★, **D6** Observabilidad de
ejecución (¿trayectoria completa visible?).

**Tier 2 — Estructurales (×2):** **D7** Modularidad/composabilidad/ecosistema;
**D5 ★**; **D8** Fidelidad test↔producción + record/replay (**usar `min`**);
**D9** Procedencia/gestión de datos y contexto; **D10** Automatización del
pipeline (CI/CD + generación).

**Tier 3 — Contextuales (×1):** **D11** Reconfiguración runtime / evolución
post-despliegue; **D12** Gobernanza, seguridad y cumplimiento (poco desarrollada
en SDR-sensor → su ausencia es *señal*, puntuar bajo, no omitir); **D13**
Costo/esfuerzo total y velocidad de iteración.

Al incluir estas, recalcula el máximo del composite y las etiquetas Tier de la
puerta de piso (la puerta cubre **todas** las dimensiones Tier 1 del set usado).

---

## 9. Reproducibilidad del propio scoring

El proceso de puntuación debe satisfacer **D1 y D2**. Cuando sea posible:
puntuar desde la rúbrica anclada de forma **independiente dos veces**, calcular
el **acuerdo** y **reconciliar** los desacuerdos contra la evidencia citada (ese
paso de reconciliación *es* un punto de control humano, D2). Así la matriz no se
convierte en opinión disfrazada de números.

---

## 10. Procedimiento (paso a paso)

1. Confirma el set de dimensiones (default D1–D5) y la disponibilidad de las
   metodologías en la entrada; si faltan, pídelas.
2. Para cada metodología y cada dimensión: identifica la faceta **SDR** y la
   faceta **AI**, ancla cada una en la rúbrica §4 y **cita la evidencia**.
3. Aplica la regla de combinación de §5 → score final por dimensión; marca 🚩 los
   desbalances.
4. Calcula composite, normalizado y la puerta de piso Tier 1 (§6).
5. Emite la salida en el formato consolidado de §7 (matriz única + hallazgos + desglose de evidencia).
6. **No lideres con el número:** resalta brechas y descalificaciones antes que el
   total.

---

*Skill derivada de `DIMENSION_MATRIX_SCORING.md`. Produce puntajes decimales
(1.0–5.0) repetibles, ligados a evidencia y comparables entre metodologías dentro
del scope indicado.*
