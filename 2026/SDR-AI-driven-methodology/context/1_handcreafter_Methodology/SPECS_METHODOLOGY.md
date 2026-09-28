# Criterios metodológicos

## Objetivo:
Diseñar un producto funcional y de calidad, que cumpla los requerimientos del cliente + normativos aplicables, mediante la aplicación de buenas prácticas de ingeniería de software, analítica de datos y gestión de proyectos
- Debe de implementar AI-assisted (importancia media)
- Específico software para SDR (crítico)

## Diseño

#### Input
- Requerimientos específicos del cliente para el producto
- Restricciones de hardware, si es posible BOM del producto con sus respectivos datasheets
- Funcionalidades generales del producto
- Funcionalidades específicas del producto
- Tipo de TRL que se logra alcanzar con el producto
- Normativas aplicables al producto (Coherente con funcionalidades y TRL)

#### Output
- Diseño del producto (full-flow), describiendo servicio por servicio, funcionalidad por funcionalidad, y la arquitectura de software y hardware del producto.
- Metodología de calibración del hardware, diseño de pruebas de laboratorio y de campo necesarias para validar el producto

## Implementación

#### Input
- Outputs del diseño del producto + documentación normativa aplicable (RAG posiblemente)
#### Output
- Producto funcional y de calidad, que cumpla los requerimientos del cliente + normativos aplicables, habiendo aplicado validaciones

## Validación

#### Input
- Producto implementado (ejecutables, firmware, hardware)
- Plan de pruebas (TDD, integración, estrés)
- Metodología de calibración (del diseño)
- Especificaciones de pruebas de laboratorio y campo

#### Output
- Reportes de pruebas unitarias e integración (cobertura ≥XX%)
- Reportes de pruebas de estrés
- Certificación HITL (funcionalidades validadas por cliente)
- Reportes de calibración y pruebas de laboratorio/campo
- Matriz de trazabilidad (Requisito → Test → Resultado ✓/✗)


#### Feedback

**Diseño**
- Día del juicio con golden reference (especificaciones cliente) vs BOM (especificaciones producto) vs diseño (especificaciones producto), si hay contradicciones volver a diseño, si no hay contradicciones pasar a implementación

**Implementación**
- Error de sintaxis (volver a implementación) [no IA]
- Código muerto (volver a implementación)
- tests unitarios o de integración fallidos (volver a implementación) [no IA]
- Ejecución de código en hardware (Si hardware no soporta el software, volver a diseño, si lo soporta pero falla volver a implementación)

**Validación**

- Checklist por feature o servicio, que indique si se cumplen los requerimientos del cliente y normativos aplicables (buscar el normativa)
- Si falla validación, volver a diseño o implementación según gravedad de error (Si error requiere refinar arquitectura, volver a diseño; si error requiere refinar implementación, volver a implementación) [por ejemplo, si falla validación de calibración, volver a diseño; si falla validación de integración, volver a implementación]

