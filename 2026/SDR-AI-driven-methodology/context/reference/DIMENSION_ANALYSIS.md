# DIMENSION_ANALYSIS.md
## Dimensiones de análisis para metodologías de desarrollo de software
### Scope: SDR como sensor/herramienta + desarrollo AI-assisted (LLMs, agentes, skills, SDD) con HITL

---

## 1. Propósito y alcance

Este documento define un marco de **dimensiones de evaluación** para analizar y comparar
metodologías de desarrollo de software cuyo alcance combina dos ejes:

- **SDR como sensor/herramienta**: software de aplicación que *consume* datos de un SDR
  (monitoreo de espectro, sensing RF, adquisición/streaming de IQ). **No** incluye el diseño
  del radio, firmware, waveforms ni cumplimiento SCA/STRS.
- **AI-assisted con HITL**: el *proceso de desarrollo* está asistido por LLMs, agentes, skills
  y prácticas tipo Spec-Driven Development (SDD), con Human-in-the-loop para revisar, aprobar
  y corregir lo que hace la IA. **No** es entrenar un modelo de ML como objetivo.

El marco sirve para dos usos:
1. **Diagnóstico**: exponer fortalezas y vacíos de una metodología concreta.
2. **Comparación**: puntuar varias metodologías sobre un eje común (ver matriz de scoring sugerida).

---

## 2. Cómo se derivaron las dimensiones

Las dimensiones se obtuvieron *traslapando* dos cuerpos de literatura y práctica:

**A) SDR-sensor / pipelines de datos en tiempo real**
Arquitecturas de monitoreo de espectro con pipelines multihilo y metadatos PDU (id de sensor,
frecuencia, timestamp); reducción de datos en el borde con presupuestos de latencia explícitos;
SigMF (Signal Metadata Format) para encapsular la grabación IQ con sus metadatos; grabación/replay
de IQ para probar decodificadores; GNU Radio / GR4 orientado a workflows reproducibles y
"AI-enabled"; y el ciclo de vida de data/streaming pipelines (prototipar → verificar semántica →
release; observabilidad y calidad de datos).

**B) Desarrollo AI-assisted con HITL**
Spec-Driven Development (spec ejecutable y versionada como única fuente de verdad, con separación
de diseño/implementación y loop de validación); Agent Development Lifecycle (observabilidad,
contención y evaluación continua por fase, dada la no-determinación); Evaluation-Driven Development
(evaluación continua de todo el sistema compuesto: contexto, planificación, memoria, guardrails);
y HITL formalizado (el humano guía plan y código y retiene control para revisar/refinar en cada
paso; métricas como tasa de override y de escalación).

**Criterio de inclusión**: cada dimensión aparece en *ambos* cuerpos y es *discriminante*
(distingue una metodología de otra), no meramente descriptiva. El mapeo se apoya en analogías
estructurales:

| Concepto SDR-sensor | Concepto AI-assisted |
|---|---|
| Spec de captura + SigMF | Spec de SDD |
| Abstracción del hardware | Abstracción del modelo/proveedor |
| Hardware-in-the-loop (HIL) | Human-in-the-loop (HITL) |
| Grabación/replay de IQ | Eval offline ↔ online |
| Metadatos SigMF/PDU | Context engineering / memoria / grounding |

---

## 3. Las 5 mejores dimensiones

Selección para *este* scope (SDR-sensor + AI-assisted + HITL). Son las más **repetidas** y con
mayor **poder diagnóstico**, y en conjunto cubren ambos dominios sin redundancia.

> Nota: **Observabilidad de ejecución** es la "sexta" más fuerte y queda embebida como el
> *habilitador* de la Dimensión 4 (sin trazas no hay evaluación). Se lista aparte en la sección 4.

---

### D1 · Reproducibilidad y determinismo
**Pregunta de evaluación:** ¿El mismo insumo produce el mismo resultado, y es auditable/regenerable?

- **Por qué es la mejor:** es el punto más frágil en *ambos* mundos, por razones **opuestas** —
  en SDR por variabilidad física del canal; en LLMs por estocasticidad del modelo. Justo por eso
  discrimina con claridad entre metodologías serias y las que no abordan el problema.
- **Ancla SDR-sensor:** grabación/replay de IQ, contenedores, calibración, congelado de semillas
  y versiones de captura.
- **Ancla AI-assisted:** control de temperatura/semilla, versionado de prompts y contexto,
  regeneración desde spec en vez de parchear.
- **Débil ↔ Fuerte:** *Débil* = resultados no repetibles, sin captura de entradas.
  *Fuerte* = escenario reproducible bit a bit (o estadísticamente acotado) con artefactos versionados.
- **Indicador sugerido:** % de ejecuciones reproducibles; existencia de dataset/seed/version pinning.

---

### D2 · Puntos de control Human-in-the-loop (HITL)
**Pregunta de evaluación:** ¿Dónde revisa, aprueba o corrige el humano, y con qué métricas se
gobierna ese lazo?

- **Por qué es la mejor:** es un requisito explícito del scope. La industria ya tiene métricas
  accionables, y el HITL es el análogo directo del HIL en SDR (validar contra señal/hardware real).
- **Ancla SDR-sensor:** operador que valida detecciones contra señal real; awareness en tiempo real.
- **Ancla AI-assisted:** el humano guía planificación y código y aprueba en cada paso; el agente
  pausa para aprobación/clarificación/revisión.
- **Débil ↔ Fuerte:** *Débil* = revisión ad hoc al final. *Fuerte* = puntos de control definidos
  por riesgo, con override/escalación medidos y umbrales de confianza calibrados.
- **Indicador sugerido:** tasa de override humano; tasa de escalación; nº de gates de aprobación.

---

### D3 · Fuente de verdad y trazabilidad
**Pregunta de evaluación:** ¿Existe un artefacto autoritativo y versionado del que se genera y
contra el que se traza todo (intención → implementación → salida)?

- **Por qué es la mejor:** separa el desarrollo disciplinado (SDD, contratos) del "vibe coding"
  o el scripting improvisado; es prerequisito de D1 y D4.
- **Ancla SDR-sensor:** spec de captura/procesamiento + SigMF como contrato de la señal.
- **Ancla AI-assisted:** spec ejecutable y versionada como single source of truth (SDD).
- **Débil ↔ Fuerte:** *Débil* = el código/flowgraph es la única verdad, sin trazabilidad.
  *Fuerte* = spec versionada, trazable requisito→artefacto→salida, regenerable.
- **Indicador sugerido:** cobertura de trazabilidad; ¿la spec está versionada y es ejecutable?

---

### D4 · Evaluación / V&V continua (con observabilidad como habilitador)
**Pregunta de evaluación:** ¿La verificación es un evento puntual o un lazo continuo a nivel de
sistema, apoyado en trazas observables?

- **Por qué es la mejor:** en sistemas no-deterministas y en pipelines de sensor en vivo, testear
  solo "al final" no basta; hay que evaluar el sistema *compuesto* (contexto, planificación,
  memoria, guardrails / o front-end, pipeline, detección) de forma continua.
- **Ancla SDR-sensor:** métricas end-to-end (probabilidad de detección, latencia), calidad de
  datos en el stream, bloque de monitoreo/feedback.
- **Ancla AI-assisted:** Evaluation-Driven Development / ADLC; el loop de mejora empieza con un
  trace; LLM-as-judge + revisión humana.
- **Débil ↔ Fuerte:** *Débil* = tests puntuales sin observabilidad. *Fuerte* = evals continuas,
  a nivel de sistema, alimentadas por trazas completas de la trayectoria.
- **Indicador sugerido:** cobertura de evals; ¿hay trazas end-to-end?; frecuencia de evaluación.

---

### D5 · Conciencia de recursos y desempeño
**Pregunta de evaluación:** ¿La metodología modela explícitamente latencia, throughput, cómputo
y costo del runtime?

- **Por qué es la mejor:** es la realidad *time-real* del SDR-sensor que las metodologías
  AI-assisted suelen ignorar; su ausencia es señal, no ruido. Cubre el eje que D1–D4 no tocan.
- **Ancla SDR-sensor:** presupuestos de latencia, cuellos de botella de I/Q (p. ej. USB),
  duty cycle, reducción de datos en el borde, throughput sostenido.
- **Ancla AI-assisted:** ventana de contexto, costo por token, latencia de inferencia,
  límites de cómputo.
- **Débil ↔ Fuerte:** *Débil* = desempeño se descubre en producción. *Fuerte* = presupuestos
  explícitos de latencia/throughput/costo, medidos desde el prototipo.
- **Indicador sugerido:** presupuesto de latencia definido; costo/operación monitoreado.

---

## 4. Lista completa de dimensiones (13)

Organizadas por tier según qué tan repetidas y diagnósticas son. Las 5 anteriores corresponden a
D1–D5 (marcadas con ★).

### Tier 1 — Núcleo (aparecen en casi todas; máximo poder diagnóstico)

| ID | Dimensión | Pregunta de evaluación |
|----|-----------|------------------------|
| D3 ★ | Fuente de verdad y trazabilidad | ¿Artefacto autoritativo y versionado, trazable intención→salida? |
| D1 ★ | Reproducibilidad y determinismo | ¿Mismo insumo → mismo resultado, auditable/regenerable? |
| D4 ★ | Evaluación/V&V continua | ¿Verificación puntual o lazo continuo a nivel de sistema? |
| D2 ★ | Puntos de control HITL | ¿Dónde revisa/aprueba/corrige el humano y con qué métricas? |
| D6 | Observabilidad de ejecución | ¿Se puede ver la trayectoria completa de lo que hizo el sistema? |

### Tier 2 — Estructurales (comunes y fuertes)

| ID | Dimensión | Pregunta de evaluación |
|----|-----------|------------------------|
| D7 | Modularidad, composabilidad y ecosistema | ¿Se ensambla de componentes probados por separado?, ¿hay registro reutilizable? |
| D5 ★ | Conciencia de recursos y desempeño | ¿Modela latencia, throughput, cómputo y costo del runtime? |
| D8 | Fidelidad test↔producción (sim-to-real / offline-online) + record/replay | ¿Qué tan fiel es el test a producción?, ¿puedes reproducir escenarios reales? |
| D9 | Procedencia y gestión de datos/contexto | ¿Cómo se captura, versiona y fundamenta el insumo? |
| D10 | Automatización del pipeline (CI/CD + generación) | ¿Qué fracción de build→test→deploy está automatizada?, ¿se generan artefactos desde la fuente de verdad? |

### Tier 3 — Contextuales (presentes según caso de uso)

| ID | Dimensión | Pregunta de evaluación |
|----|-----------|------------------------|
| D11 | Reconfiguración runtime / evolución post-despliegue | ¿Cambia comportamiento sin redeploy completo? |
| D12 | Gobernanza, seguridad y cumplimiento | ¿Guardrails, control de acceso, cumplimiento regulatorio? |
| D13 | Costo/esfuerzo total y velocidad de iteración | ¿Reduce costo (dev+operación) y latencia idea→prototipo verificable? |

---

## 5. Mapa de traslape (dónde se repite cada dimensión)

| ID | Dimensión | Ancla SDR-sensor | Ancla AI-assisted / HITL | Tier |
|----|-----------|------------------|--------------------------|------|
| D1 | Reproducibilidad / determinismo | IQ record/replay, contenedores, calibración | No-determinismo LLM, seed/version pinning | 1 |
| D2 | HITL control points | HIL vs. señal real, operador | Guía plan+código, override/escalation rate | 1 |
| D3 | Fuente de verdad + trazabilidad | Spec de captura + SigMF | SDD spec autoritativa versionada | 1 |
| D4 | Evaluación/V&V continua | Métricas end-to-end, calidad en stream | EDD / ADLC, LLM-as-judge + humano | 1 |
| D6 | Observabilidad de ejecución | Bloque monitoring/feedback, métricas RT | Traces → evals | 1 |
| D7 | Modularidad + ecosistema | Bloques/OOT de GNU Radio, capas | Skills, tools, sub-agentes, frameworks | 2 |
| D5 | Recursos / desempeño | Latencia, throughput, duty cycle, edge | Context window, costo/token, latencia | 2 |
| D8 | Fidelidad test↔prod | Replay vs. live, sim↔experimento | Offline↔online, spec-to-note gap | 2 |
| D9 | Procedencia datos/contexto | Metadatos SigMF/PDU | Context engineering, memoria, grounding | 2 |
| D10 | Automatización pipeline | Release workflow, flowgraphs | Generación de código/tests desde spec | 2 |
| D11 | Reconfiguración runtime | Cambio de banda por software | Corrección post-deploy sin tocar código | 3 |
| D12 | Gobernanza / seguridad | Regulación de espectro, uso indebido | Guardrails, EU AI Act (supervisión humana) | 3 |
| D13 | Costo + velocidad de iteración | COTS, rapid prototyping | Agentes autónomos más rápidos/baratos | 3 |

---

## 6. Tensiones a vigilar (lectura crítica)

- **D1, D2 y D8** son las que más *tensión* generan al cruzar ambos mundos: el SDR-sensor asume
  pipelines deterministas, mientras que los LLMs no lo son. Son las más útiles para exponer
  debilidades de una metodología candidata.
- **D12 (gobernanza)** está poco desarrollada en la literatura SDR-sensor pero es central en la
  AI-assisted. Su ausencia en una metodología es *señal*, no ruido.
- **D5 (recursos)** es lo inverso: madura en SDR-sensor, frecuentemente ignorada en metodologías
  AI-assisted. Una metodología unificada fuerte debe cubrir ambos.

---

## 7. Uso sugerido como matriz de scoring

Para comparar metodologías, puntuar cada dimensión de **1 a 5** y ponderar por tier
(sugerido: Tier 1 ×3, Tier 2 ×2, Tier 3 ×1). Un gráfico de radar por metodología facilita
la comparación visual.

| Dimensión | Peso (tier) | Metodología A | Metodología B | ... |
|-----------|-------------|---------------|---------------|-----|
| D1–D13 | ×3 / ×2 / ×1 | 1–5 | 1–5 | ... |

---

## 8. Referencias (tipos de fuente)

Síntesis a partir de: Spec-Driven Development (Thoughtworks, Microsoft, GitHub Spec Kit);
Evaluation-Driven Development de agentes LLM; Agent Development Lifecycle (IBM, LangChain);
HITL para agentes de desarrollo (HULA/Atlassian) y oversight de agentes; SigMF y toolkits de
grabación IQ; arquitecturas de monitoreo de espectro con SDR de bajo costo; GNU Radio / GR4;
y prácticas de ciclo de vida de pipelines de datos/streaming (Google SRE, data engineering
lifecycle).

*Documento generado como marco de análisis; las dimensiones son abstracciones diseñadas para
aplicarse a metodologías adicionales dentro del scope indicado.*