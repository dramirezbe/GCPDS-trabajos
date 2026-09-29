# Metodología de Desarrollo de Producto SDR — *AI-agéntica con HITL*

*Versión 2 — corregida: sin concepto de «Golden Reference» (GR); modelo de **ground truth** reforzado y desdoblado en dos fuentes de verdad (especificación vs. evidencia).*

**Alcance:** Creación de un producto basado en *Software Defined Radio* (SDR) hasta un nivel de madurez tecnológica (TRL) objetivo, cubriendo desarrollo de software, calibración de hardware y pruebas de campo, con cumplimiento simultáneo de normativa aplicable, requerimientos del cliente y restricciones del hardware empleado.

**Enfoque:** el motor principal del proceso son **agentes de IA**, que planean, generan y corrigen artefactos a lo largo del ciclo. La IA opera sobre una **capa determinista** que produce la **evidencia objetiva** (compilación, ejecución, pruebas, medición) y bajo una **capa humana** que conserva la autoridad de decisión en los puntos críticos (*human-in-the-loop*, HITL). La validación es **dirigida por especificación** (*specification-driven*): se contrasta la evidencia objetiva contra el *ground truth* autorizado por humano, **no** contra una implementación previa de referencia.

---

## 1. Objetivo

Diseñar, implementar y validar un producto SDR funcional y de calidad que:

- Cumpla los **requerimientos del cliente** (funcionales y de desempeño de RF).
- Cumpla la **normativa aplicable** (espectro, EMC, seguridad, sectoriales).
- Sea **coherente con el hardware disponible** (BOM y datasheets).
- Alcance el **TRL objetivo** definido para el proyecto.

Todo mediante un flujo **agéntico verification-first + HITL**, en el que la IA hace el trabajo de razonamiento y generación, la instrumentación determinista genera la evidencia, y el humano decide donde el riesgo lo exige.

### Restricciones del proceso

| Restricción | Criticidad | Nota |
|---|---|---|
| IA agéntica como motor del ciclo (con HITL) | **Crítica** | Define el modo de trabajo (§3–§5) |
| Software especializado para SDR (USRP / UHD / GNU Radio) | **Crítica** | Define el stack de implementación y validación |
| Cumplimiento normativo verificable | **Crítica** | Condición de liberación; sign-off humano obligatorio |

---

## 2. Principio rector

> La adopción de IA generativa no depende de su capacidad bruta sino de cuán bien se integre con las prácticas de desarrollo establecidas; la evidencia converge en un modelo **verification-first, human-in-the-loop** que rediseña el proceso completo en vez de solo automatizar subtareas [16].

De ahí se derivan dos reglas que gobiernan toda la metodología:

1. **Desarrollo dirigido por especificación (*specification-driven*):** las especificaciones —modelos de dominio, casos de uso, reglas de negocio y restricciones— las autoriza el humano y sirven como *ground truth* que los agentes implementan; esto reduce errores por alucinación y mantiene trazabilidad entre intención y salida [16]. En nuestro caso el *ground truth de especificación* son: **requerimientos del cliente + BOM/datasheets + normativa aplicable + TRL objetivo**.

2. **La verificación no la juzga la IA sobre sí misma:** el fundamento es análisis determinista, repetible y explicable, aumentado —no reemplazado— por técnicas de IA [18]. Compilar, ejecutar, medir y correr pruebas son actos deterministas que producen el *ground truth de evidencia*; la IA los invoca y los interpreta, pero no los sustituye.

### 2.1 Terminología y fuentes de verdad (lectura obligatoria)

Esta metodología distingue **dos tipos de *ground truth***. Confundirlos es la fuente más común de errores conceptuales, por lo que se definen de forma explícita.

| Término | Qué es | Quién lo produce | Rol en V&V |
|---|---|---|---|
| **Ground truth de especificación** (lo que *debe* ser) | Requisitos del cliente + BOM/datasheets + normativa aplicable + TRL objetivo | **Humano** (autoría y autorización) | Define el criterio de correcto. Es la intención contra la que se valida. |
| **Ground truth de evidencia** (lo que *es*) | Resultado de compilar, ejecutar, correr pruebas y **medir** con instrumentos trazables | **Capa determinista** (herramientas e instrumentos, no IA) | Es la observación objetiva y reproducible del comportamiento real. |

**Validación en esta metodología** = contrastar el **ground truth de evidencia** (lo que el sistema realmente hace) contra el **ground truth de especificación** (lo que debe hacer). La IA interpreta y enruta; el humano decide en los gates críticos.

**Términos que NO se usan en esta metodología:**

- **Golden Reference (GR):** implementación previa considerada «correcta» contra la que se comparan versiones nuevas (enfoque *reference-driven*). **No se emplea.** Aquí no existe una implementación anterior canónica que sirva de patrón; el patrón es la **especificación** (requisitos + BOM + normativa + TRL), y la comprobación se hace contra la **evidencia determinista**. Sustituir la especificación por una implementación previa reintroduciría el riesgo de perpetuar errores heredados y rompería la trazabilidad *intención → salida* [16].
- **Validación *reference-driven*** (contra una versión/implementación anterior). Se sustituye por validación *specification-driven* [16].

> **Aclaración sobre la palabra «referencia».** Cuando en este documento se usa «equipos de referencia» (§12) se alude a **instrumentos de medición con trazabilidad metrológica**, no a una Golden Reference de software. Son conceptos distintos: uno es un patrón de medición físico; el otro sería un patrón de implementación —que aquí no se usa—.

---

## 3. Modelo de tres capas

El proceso se organiza en tres capas con responsabilidades distintas. La clave de la metodología es **qué se delega a cada capa** y **qué fuente de verdad gobierna cada decisión**.

```
┌─────────────────────────────────────────────────────────────────────┐
│  CAPA HUMANA (HITL) — autoridad de decisión                          │
│  Autoría del GROUND TRUTH DE ESPECIFICACIÓN · sign-off de gates      │
│  críticos · conformidad normativa · aceptación de calibración y      │
│  campo · aceptación del cliente · seguridad                          │
└───────────────────────────────▲─────────────────────────────────────┘
                                 │ propone / escala          │ aprueba / veta
┌───────────────────────────────┴─────────────────────────────────────┐
│  CAPA DE AGENTES DE IA (motor principal)                             │
│  Planear · generar y corregir código · explorar diseño · analizar    │
│  coherencia req↔BOM↔normativa · clasificar fallos y enrutar ·        │
│  RAG normativo · generar casos de prueba · mantener RTM · redactar   │
│  → CONTRASTA evidencia (abajo) contra especificación (arriba)        │
└───────────────────────────────▲─────────────────────────────────────┘
                                 │ invoca      │ GROUND TRUTH DE EVIDENCIA
┌───────────────────────────────┴─────────────────────────────────────┐
│  CAPA DETERMINISTA — produce el GROUND TRUTH DE EVIDENCIA (NO IA)    │
│  Compilador · runner de tests · análisis estático · linter/formato · │
│  ejecución en hardware · instrumentos de medición y calibración      │
└─────────────────────────────────────────────────────────────────────┘
```

> **Dos verdades, un contraste.** La capa humana fija *qué debe ser* (especificación); la capa determinista observa *qué es* (evidencia). La capa de IA es el motor que las confronta y propone acciones. No hay una tercera «implementación de referencia» en el medio.

### 3.1 Capa determinista — produce el *ground truth de evidencia* (lo que **NO** hace la IA)

Genera evidencia objetiva y reproducible. La IA la consume pero nunca la reemplaza ni la «opina».

- Compilación y *build*.
- Ejecución de código (en simulación y sobre el hardware objetivo).
- Ejecución de pruebas (unitarias, integración, estrés).
- Análisis estático, *linting* y formateo automático.
- Detección de errores de sintaxis.
- Medición física en calibración y en RF (instrumentos con trazabilidad metrológica).

> El análisis determinista se integra dentro del propio bucle de generación del agente, dándole retroalimentación en tiempo real sobre riesgos y errores; esto es distinto de un LLM revisándose a sí mismo, que no es consistente ni explicable [18].

### 3.2 Capa de agentes de IA (motor principal)

Es el actor por defecto del ciclo. Un agente de codificación opera en un bucle de razonamiento: recibe la tarea, la descompone, genera código, evalúa su salida contra la capa determinista, interpreta los mensajes de error y ajusta su enfoque a lo largo de varios ciclos antes de presentar un resultado [18]. Tareas donde la IA es primaria:

- **Creación y corrección de código** SDR (bloques GNU Radio/UHD, firmware, scripts de prueba).
- **Exploración y propuesta de diseño** y de particionamiento HW/SW.
- **Análisis de coherencia** requisitos ↔ BOM ↔ normativa ↔ diseño (insumo del Gate 1).
- **Contraste evidencia ↔ especificación:** confrontar el *ground truth de evidencia* contra el *ground truth de especificación* y reportar desviaciones.
- **Clasificación de fallos** por causa raíz y **enrutamiento** a la fase correcta.
- **RAG normativo:** recuperar la norma aplicable y mapear Norma → Requisito → Test.
- **Generación de casos de prueba** y mantenimiento de la **RTM**.
- **Redacción** de documentación técnica y de reportes.
- **Pre-decisión de gates** (LLM-as-Judge) que el humano confirma o veta.

> Ganancias realistas: la evidencia disponible respalda mejoras modestas, del orden de 2–3× en tareas acotadas y bien especificadas [19]; los agentes generan código más rápido de lo que los equipos alcanzan a verificarlo —el llamado *"verification tax"*— por lo que la capa de verificación y el HITL son parte esencial del diseño, no un añadido [20].

### 3.3 Capa humana (HITL — autoridad)

El rol humano se desplaza hacia arquitectura, planeación y revisión, dedicando menos tiempo a escribir código línea por línea y más a asegurar que lo que producen los agentes cumpla los estándares [18]. El humano conserva **control total para revisar y refinar** las salidas en cada paso [21], y el flujo se **pausa en checkpoints estratégicos** para validación y retroalimentación humana antes de continuar [17]. Decisiones reservadas al humano:

- **Autoría y autorización del *ground truth de especificación*** (requisitos, BOM, normativa, TRL).
- Sign-off de gates críticos (Gate 1 para *features* críticas; Gate 3 siempre).
- Conformidad normativa.
- Aceptación de calibración y de resultados de campo.
- Aceptación del cliente (HITL de producto).
- Cualquier asunto de seguridad.

---

## 4. Autonomía graduada de la IA

Regla general:

> **La autonomía del agente es inversamente proporcional a (irreversibilidad × riesgo normativo/seguridad).**
> Cuanta más evidencia determinista y más reversible sea la acción, más autónomo puede ser el agente. Cuanto más pese la normativa, el campo o el cliente, más obligatorio es el HITL.

| Nivel | Quién decide | Cuándo aplica | Ejemplo en este proyecto |
|---|---|---|---|
| **A1 – Autónomo** | Agente (humano puede vetar *a posteriori*) | Evidencia determinista + acción reversible | Corregir un test en rojo, refactor, formateo, resolver error de compilación |
| **A2 – Autónomo con revisión** | Agente decide, humano revisa por muestreo | Riesgo bajo, evidencia objetiva | Gate 2 (código y *build*); enrutar un fallo de integración a Implementación |
| **A3 – Propuesta + sign-off** | Agente propone, humano aprueba | Riesgo medio o impacto en arquitectura | Gate 1 (coherencia de diseño) para *features* que tocan normativa o TRL |
| **A4 – HITL obligatorio** | Humano decide, agente asiste | Alto riesgo / irreversible / normativo | Gate 3, aceptación de calibración, pruebas de campo, aceptación de cliente |

---

## 5. Roles de agentes

Se recomienda un pequeño «equipo» de agentes coordinados (implementable, p. ej., como un grafo de estados con puntos de interrupción para HITL [17]):

| Agente | Función | Autonomía típica |
|---|---|---|
| **Orquestador** | Coordina el flujo entre fases y decide cuándo pausar para HITL | A2–A3 |
| **Diseño** | Propone arquitectura y particionamiento HW/SW; analiza coherencia req↔BOM↔normativa | A3 |
| **Codificación** | Genera y corrige código en bucle contra la capa determinista | A1–A2 |
| **QA/Validación** | Diseña y ejecuta pruebas, contrasta evidencia ↔ especificación, resume evidencia, propone veredicto de gate | A2 (Gate 2) / A4 (Gate 3) |
| **Normativo (RAG)** | Recupera normas, construye el mapeo Norma→Requisito→Test | A3 (siempre con sign-off humano de conformidad) |
| **Juez (LLM-as-Judge)** | Pre-evalúa gates; se combina con revisión humana | A3–A4 |

> Los agentes deben evaluarse contra *benchmarks* y *policy checks* que reflejen el comportamiento esperado, combinando **LLM-as-Judge** con revisión **human-in-the-loop**: escala del primero, confianza del segundo [22].

---

## 6. Fundamento metodológico

| Elemento de esta metodología | Marco / estándar de referencia |
|---|---|
| Estructura V&V (verificar en diseño, validar en producto) | V&V en SDLC; V&V en investigación de diseño [1][2] |
| Gates formales de diseño (Gate 1 / Gate 3) | PDR / CDR / TRR (NASA, aeroespacial, DoD 5000) [5][6] |
| Diseño conjunto hardware/software | Hardware-Software Codesign (IEEE) [3][4] |
| Stack SDR | USRP + GNU Radio + UHD (Ettus Research) [7][8] |
| Nivel de madurez objetivo | Technology Readiness Levels (NASA / DoD) [13][14] |
| Trazabilidad requisito → test → resultado | Requirements Traceability Matrix (RTM) [10][15] |
| **Ground truth de especificación / validación specification-driven** | **Agentic SDLC specification-driven; ground truth y policy checks** [16][18][22] |
| **IA agéntica + HITL + verification-first** | **Agentic SDLC / HITL** [16][17][18][19][20][21][22] |

---

## 7. Estructura general

```
                 ┌──────────────────────────────────────────────┐
                 │  GROUND TRUTH DE ESPECIFICACIÓN (autoría hum.) │
                 │  Req. cliente · BOM+datasheets · Normativa    │
                 │  TRL objetivo · Funcionalidades              │
                 └──────────────────────┬───────────────────────┘
                                        ▼
        ┌───────────────┐  Gate 1 (A3) ┌────────────────┐ Gate 2 (A2) ┌───────────────┐
        │    DISEÑO     │ ─(PDR+HITL)─▶│ IMPLEMENTACIÓN │ ─(build)──▶ │  VALIDACIÓN   │
        │  agente:Diseño│              │agente:Codificac│             │agente:QA      │
        └───────┬───────┘              └────────┬───────┘             └───────┬───────┘
                ▲                               ▲                             │ Gate 3 (A4)
                │      error de arquitectura    │  error de código           │ (CDR/TRR+HITL)
                └───────────────────────────────┴─────────────────────────────┘
                        CICLO AGÉNTICO  (agentes en el bucle · humano en los checkpoints)
                     evidencia determinista ── contrastada contra ──▶ especificación
                                        │
                                        ▼
                              ┌──────────────────┐
                              │   LIBERACIÓN     │  (cumple cliente + normativa + TRL)
                              └──────────────────┘
```

El bucle agéntico ejecuta planear → implementar → probar y alimenta los resultados de verificación a la siguiente iteración **entre checkpoints humanos** [19]. Clasificar los hallazgos por causa raíz y devolverlos a la fase correspondiente (arquitectura → Diseño; código → Implementación) es coherente con el codiseño HW/SW, donde cada fase considera de forma conjunta restricciones físicas del hardware y del software [3].

---

## 8. Fase: Diseño

**Motor:** agente de Diseño (propone) · **Autoridad:** HITL en Gate 1 para *features* críticas.

### Entradas (*ground truth de especificación*, autoría humana)
- Requerimientos específicos del cliente.
- Restricciones de hardware — de ser posible, **BOM con datasheets**.
- Funcionalidades generales y específicas.
- TRL objetivo alcanzable.
- Normativa aplicable (coherente con funcionalidades y TRL).

### Salidas
- Diseño *full-flow*: servicio por servicio, funcionalidad por funcionalidad, arquitectura SW/HW.
- **Metodología de calibración** del hardware (§12).
- **Diseño de pruebas** de laboratorio y de campo (§13).
- Matriz de trazabilidad preliminar (Requisito → Caso de prueba).

> El diseño parte de una especificación a nivel de sistema y realiza particionamiento HW/SW, síntesis de comunicaciones y generación de arquitectura, guiado por desempeño y restricciones del contexto [4]. El agente de Diseño automatiza esta exploración; el humano fija la intención y revisa el resultado.

### Gate 1 — Revisión de diseño (tipo PDR) · Autonomía A3

El agente analiza la coherencia entre las tres fuentes del *ground truth de especificación* y **propone** un veredicto; el humano lo aprueba (obligatorio si la *feature* toca normativa o TRL).

| Se compara | Contra requisitos cliente | Contra BOM | Contra diseño |
|---|---|---|---|
| **Requisitos cliente** *(ground truth de especificación)* | = | ¿El HW los soporta? | ¿El diseño los satisface? |
| **BOM** | ¿Soporta funcionalidades? | = | ¿Diseño viable con este HW? |
| **Diseño** | ¿Los satisface? | ¿Es realizable? | = |

**Criterios de aceptación:** cero contradicciones; 100 % de requisitos mapeados al diseño; HW soporta todas las funcionalidades; normativa considerada en la arquitectura.

**Decisión:** contradicciones → volver a **Diseño**; sin contradicciones y con sign-off → **Implementación**.

> Un PDR comprueba temprano, cuando los cambios aún son baratos, si la arquitectura elegida puede cumplir los requisitos [5]; los gates son controles de riesgo con criterios de entrada, evidencia auditable y criterios de salida, y pasarlos autoriza a comprometer más recursos [6].

---

## 9. Fase: Implementación

**Motor:** agente de Codificación en bucle contra la capa determinista · **Autoridad:** revisión por muestreo.

### Entradas
- Salidas del Diseño; normativa aplicable (vía RAG); plan de V&V + RTM.

### Salidas
- Código fuente SDR (bloques GNU Radio/UHD), firmware, hardware ensamblado según BOM.
- Documentación técnica (interfaces, protocolos, configuración).

> El stack se apoya en el ecosistema estándar de SDR: todos los USRP soportan GNU Radio, y la mayoría soporta RFNoC para desarrollo FPGA sin escribir VHDL/Verilog [7]; el control del hardware se hace vía UHD, que transporta las muestras de RF y ajusta frecuencia, tasa de muestreo y ganancias [8].

### Gate 2 — Validación de código y *build* · Autonomía A2 (el agente decide)

La **capa determinista** genera la evidencia (*ground truth de evidencia*); el **agente** la interpreta, la contrasta contra la especificación, clasifica el fallo, corrige y re-ejecuta; el humano revisa por muestreo.

| Hallazgo | Origen de la evidencia | Quién actúa | Regresa a |
|---|---|---|---|
| Error de sintaxis / compilación | Compilador (determinista) | Agente corrige | **Implementación** |
| Código muerto | Análisis estático (determinista) | Agente limpia | **Implementación** |
| Tests unitarios / integración fallidos | Runner de tests (determinista) | Agente depura | **Implementación** |
| Formato / estilo | Linter/formatter (determinista) | Auto-fix | **Implementación** |
| **HW no soporta el SW** | Ejecución en hardware (determinista) | Agente escala | **Diseño** (A3: sign-off) |
| SW falla sobre el HW | Ejecución en hardware (determinista) | Agente depura | **Implementación** |

**Regla:** la IA nunca «aprueba a ojo» un error de sintaxis, una ejecución o un test; esos veredictos vienen de la herramienta determinista. La IA aporta el **diagnóstico, la corrección y el enrutamiento**, y el **contraste contra la especificación** (no contra una implementación previa).

---

## 10. Fase: Validación

**Motor:** agente de QA (ensambla evidencia y propone) · **Autoridad:** HITL obligatorio (Gate 3).

### Entradas
- Producto implementado; plan de pruebas (TDD, integración, estrés); metodología de calibración; BOM + datasheets; especificaciones de laboratorio y campo.

### Salidas
- Reportes de pruebas unitarias e integración (cobertura ≥ objetivo).
- Reportes de estrés (acotados a BOM/datasheets).
- Reportes de calibración y de laboratorio/campo.
- **Certificación HITL** (funcionalidades validadas por el cliente).
- **RTM completa** (Requisito → Test → Resultado ✓/✗).
- Certificado de cumplimiento normativo.

### Actividades
1. **Pruebas de software:** unitarias e integración (TDD) + estrés acotado al HW real del BOM. *(Ejecución determinista; el agente diseña, interpreta y contrasta contra la especificación.)*
2. **Calibración:** ejecución de la metodología del Diseño, validada contra estándares y equipos de referencia metrológica (§12).
3. **Laboratorio y campo:** según §13.
4. **QA por funcionalidad (HITL):** cada servicio se acepta con humano en el bucle.
5. **Checklist normativo:** el agente Normativo recupera la norma (RAG) y mapea Norma→Requisito→Test; **la conformidad la firma el humano**.

### Gate 3 — Revisión crítica y de resultados (tipo CDR/TRR) · Autonomía A4 (HITL)

El agente de QA **resume la evidencia y propone**; la decisión de liberar es humana. El veredicto se emite comparando la evidencia determinista contra el *ground truth de especificación*, no contra una versión previa del producto.

> Un CDR es el paso completo sobre el diseño terminado y su análisis antes del sign-off previo a despliegue [5]; el TRR verifica que planes, equipos y artículos de prueba están listos, y el FDR evalúa los resultados reales y resuelve anomalías antes de entregar a producción [9].

**Clasificación de fallos de validación (propuesta por el agente, confirmada por el humano):**

| Tipo | Ejemplo | Regresa a |
|---|---|---|
| Arquitectónico | Falla la calibración; el diseño no soporta un requisito normativo | **Diseño** |
| Implementación | Falla la integración; bug; protocolo mal implementado | **Implementación** |
| Especificación | El cliente rechaza una funcionalidad (HITL); requisito ambiguo o contradictorio en el *ground truth de especificación* | **Diseño** (renegociar) |

**Ejemplos:** falla de calibración → **Diseño**; falla de integración → **Implementación**.

---

## 11. Retroalimentación y gestión de desviaciones

```
Fallo detectado (ground truth de evidencia, determinista)
     ▼
Agente: contraste evidencia↔especificación + análisis de causa raíz
        + clasificación + propuesta de enrutamiento
     ▼
¿Nivel de autonomía?  ──A1/A2──▶ agente corrige y re-valida
     │
     A3/A4 ──▶ checkpoint HITL (humano aprueba/veta)
     ▼
Corregir en la fase correspondiente → re-validar
     ▼
¿Aprobado?  ── Sí ──▶ siguiente feature / preparar liberación
     │
     No ──▶ repetir ciclo (máx. 3 iteraciones por feature)
```

**Control de calidad del proceso:**
- Máximo 3 ciclos de re-validación por *feature*; si persiste, **escalar a humano** (arquitecto + PM).
- Todo lo que sea A3/A4 pasa por checkpoint humano antes de comprometer cambios.
- Documentar desviaciones y lecciones aprendidas (el agente redacta, el humano valida).

Esta bidireccionalidad —de requisito a test y de test a requisito— es la función clave de la RTM: cobertura completa, resultados ligados a objetivos y actualización ágil cuando cambian los requisitos [10].

---

## 12. Metodología de calibración (SDR)

Se **diseña** en Diseño y se **ejecuta** en Validación. El agente planifica e interpreta; **la medición es determinista y la aceptación es humana (A4)**.

- **Parámetros de RF:** frecuencia central, ganancias RX/TX, tasa de muestreo, potencia de salida, offsets DC/IQ, dentro del rango del HW del BOM.
- **Equipos de referencia metrológica** con trazabilidad (patrón físico de medición; **no** una Golden Reference de software — ver §2.1).
- **Criterios de aceptación** (tolerancias e incertidumbres) derivados de requisitos del cliente y normativa.
- **Procedimiento repetible** y registrable.

El control de estos parámetros se ejecuta vía UHD, que provee acceso para transportar muestras de RF y controlar frecuencia, tasa de muestreo y amplificaciones [8]. Los valores medidos son *ground truth de evidencia*; el agente puede proponer ajustes, pero la conformidad de calibración la acepta el humano.

---

## 13. Pruebas de laboratorio y de campo (SDR)

- **Laboratorio:** entorno controlado, incluyendo estrés acotado a las capacidades del hardware.
- **Campo:** condiciones operativas reales, necesarias para TRL altos. **Aceptación A4 (HITL).**

> USRP + GNU Radio es una herramienta consolidada para desarrollar y ensayar sistemas de comunicación y realizar experimentos [11], con múltiples implementaciones publicadas que validan sobre hardware real (p. ej. B210) [12].

**Relación con TRL:** TRL 4→6 corresponde a experimentos con similitud creciente a la aplicación final, y TRL 7→9 a operar el prototipo en un rango más amplio de condiciones hasta igualar el sistema real [13]; el TRL del sistema lo fija el componente de **menor** TRL [14], por lo que el campo debe cubrir los subsistemas críticos, no solo el software.

---

## 14. Trazabilidad, aceptación y reparto IA / herramienta / humano

### 14.1 Quién hace qué (resumen)

| Actividad | Capa determinista (evidencia, no IA) | Agente de IA (motor) | Humano (HITL) |
|---|---|---|---|
| Autoría del *ground truth de especificación* (requisitos, BOM, normativa, TRL) | — | Asiste | **Autoriza** |
| Exploración y propuesta de arquitectura | — | **Propone** | Aprueba (Gate 1) |
| Coherencia req↔BOM↔normativa | — | **Analiza** | Confirma |
| Generación y corrección de código | Compila/ejecuta/testea | **Genera y corrige** | Revisa por muestreo |
| Errores de sintaxis / formato | **Detecta/decide** | Corrige | — |
| Ejecución en hardware | **Ejecuta/mide** | Interpreta | — |
| Contraste evidencia ↔ especificación | Aporta evidencia | **Contrasta y reporta** | Confirma en gates |
| Clasificación y enrutamiento de fallos | Aporta evidencia | **Clasifica y enruta** | Confirma si A3/A4 |
| Decisión de Gate 2 | Aporta evidencia | **Decide (A2)** | Revisa por muestreo |
| Decisión de Gate 1 | — | Propone | **Decide (A3)** |
| Decisión de Gate 3 / liberación | Aporta evidencia | Propone | **Decide (A4)** |
| Calibración | **Mide** | Planifica/interpreta | **Acepta** |
| Pruebas de campo | **Ejecuta/mide** | Analiza | **Acepta** |
| Conformidad normativa | — | Mapea (RAG) | **Firma** |
| Aceptación del cliente | — | Prepara evidencia | **Acepta** |
| RTM y documentación | Aporta resultados | **Mantiene/redacta** | Valida |

### 14.2 Criterios de aceptación para liberación
- [ ] 100 % de requisitos del cliente validados y aprobados (HITL).
- [ ] 100 % de normativa aplicable cumplida, con conformidad firmada.
- [ ] Cobertura de pruebas ≥ objetivo definido.
- [ ] Calibración dentro de tolerancias, aceptada contra estándares metrológicos.
- [ ] Pruebas de campo superadas para el TRL objetivo.
- [ ] RTM completa (Requisito → Test → Resultado), validada contra el *ground truth de especificación*.
- [ ] Cero no-conformidades abiertas (o *waived* formalmente por el cliente).
- [ ] Documentación técnica y certificados completos.

---

## 15. Referencias

1. *Demystifying verification and validation in design research methodology* (2026). https://www.tandfonline.com/doi/full/10.1080/09544828.2026.2624355
2. *Role of Verification and Validation (V&V) in SDLC* — GeeksforGeeks. https://www.geeksforgeeks.org/software-testing/role-of-verification-and-validation-vv-in-sdlc/
3. *Hardware-software codesign of embedded systems*. https://www.academia.edu/2739034/Hardware_software_codesign_of_embedded_systems
4. Abid, M. et al. *Hardware/Software Co-Design Methodology for Design of Embedded Systems* (1998). https://journals.sagepub.com/doi/10.3233/ICA-1998-5106
5. *What is PDR and CDR?* — CoLab. https://www.colabsoftware.com/faqs/what-is-pdr-and-cdr
6. *PDR, CDR, MRR… Space Project Milestones Decoded* (2025). https://anywaves.com/resources/blog/space-project-milestones-a-practical-guide/
7. *SDR Software* — Ettus Research. https://www.ettus.com/sdr-software/
8. *Location Information Sharing Using SDR in Multi-UAV Systems* (2025) — descripción de UHD. https://arxiv.org/pdf/2506.17678
9. *Design Review: Types, Process, and DFSS Tollgate Connection*. https://sixsigmadsi.com/design-review/
10. *Requirements Traceability Matrix (RTM) for Systems Engineers* — ReqView. https://www.reqview.com/blog/requirements-traceability-matrix/
11. *Review of Design and Implementation of New Software Radio System based on GNU Radio and USRP*. https://www.researchgate.net/publication/390960689
12. *Non-Orthogonal HARQ-CC over SDR: A GNU Radio-Based Implementation* (2026). https://arxiv.org/pdf/2603.03746
13. *A Sociotechnical Readiness Level Framework* (2024) — escala TRL. https://arxiv.org/pdf/2403.18204
14. *Technology readiness level* — Wikipedia (síntesis NASA/DoD). https://en.wikipedia.org/wiki/Technology_readiness_level
15. *Requirements Traceability Matrix: Your QA Strategy* — Abstracta (2025). https://abstracta.us/blog/testing-strategy/requirements-traceability-matrix-your-qa-strategy
16. *Rethinking Software Engineering for Agentic AI Systems* (2026) — verification-first + HITL, specification-driven, trazabilidad intención→salida. https://arxiv.org/pdf/2604.10599
17. Saini, G. *Agentic AI for SDLC: Automating Software Delivery with LangGraph and Human-in-the-Loop* (2025). https://medium.com/@gauravsaini.728/agentic-ai-for-sdlc-automating-software-delivery-with-langgraph-and-human-in-the-loop-e6549f92d9be
18. *What is Agentic SDLC* — Sonar (2026) — análisis determinista como fundamento; bucle de razonamiento del agente. https://www.sonarsource.com/resources/library/what-is-agentic-sdlc/
19. *The Agent-Run Loop: Reframing the SDLC as a Continuous Cycle* — Augment Code (2026). https://www.augmentcode.com/guides/agent-run-development-loop
20. *A guide to the agentic software development lifecycle (SDLC)* — CodeRabbit (2026) — *verification tax*. https://www.coderabbit.ai/guides/agentic-sdlc
21. *Human-In-The-Loop Software Development Agents: Challenges and Future Directions* (HULA, Atlassian) (2025). https://arxiv.org/pdf/2506.11009
22. *What Is the Agent Development Lifecycle?* — IBM (2026) — LLM-as-Judge + HITL, ground truth y policy checks. https://www.ibm.com/think/topics/agent-development-lifecycle-adlc

---

## Anexo A — Registro de cambios respecto a la versión anterior

| # | Cambio | Sección(es) | Motivo |
|---|---|---|---|
| 1 | Se añade **§2.1 Terminología y fuentes de verdad** con tabla de los dos *ground truth* y exclusión explícita de «Golden Reference (GR)» y de la validación *reference-driven*. | §2.1 (nueva) | Eliminar ambigüedad y dejar constancia de que GR no aplica en este enfoque *specification-driven* [16]. |
| 2 | Se **desdobla el concepto de *ground truth*** en *ground truth de especificación* (lo que debe ser, autoría humana) y *ground truth de evidencia* (lo que es, capa determinista). | §2, §3, §3.1, §3.2, §7, §12, §14 | El documento anterior llamaba «ground truth» tanto a las entradas como a la capa determinista, mezclando intención y observación. |
| 3 | Diagramas de tres capas y de estructura general reetiquetados para reflejar las dos verdades y el **contraste evidencia ↔ especificación**. | §3, §7 | Coherencia visual con la nueva terminología. |
| 4 | Etiqueta «(referencia)» en la tabla del Gate 1 cambiada a **«(ground truth de especificación)»**. | §8 | La palabra «referencia» inducía a leerla como Golden Reference. |
| 5 | «Equipos de referencia» aclarados como **equipos de referencia metrológica** (patrón físico de medición), distinguidos explícitamente de una GR de software. | §2.1, §12, §14.2 | Evitar confusión entre patrón de medición e implementación de referencia. |
| 6 | Se explicita que los veredictos de Gate 2 y Gate 3 se emiten **contra la especificación, no contra una versión previa** del producto. | §9, §10 | Reforzar el enfoque *specification-driven*. |
| 7 | Nueva actividad/fila «**Contraste evidencia ↔ especificación**» en roles y en el reparto de responsabilidades. | §3.2, §5, §14.1 | Hacer visible la operación que reemplaza al «comparar contra GR». |

---

*Documento de criterios metodológicos — versión 2, AI-agéntica con HITL, corregida: sin Golden Reference y con modelo de ground truth desdoblado (especificación vs. evidencia). Ajustada al caso de desarrollo de producto SDR con TRL objetivo, software, calibración y pruebas de campo, bajo normativa, requerimientos del cliente y restricciones de hardware.*