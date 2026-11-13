# NASA FLEX: Controlled Fuel-Input Comparison

## Experimental conditions

- Ambient pressure; mmHg: 759.6
- O initial ambient composition; mole fraction: 0.2
- CO initial ambient composition; mole fraction: 0.0
- He initial ambient composition; mole fraction: 0.0
- Droplet initial diameter; mm: 3.01

## Prediction results

### Heptane

Predicted outcome: **Disruption**

- Disruption: 0.431665 (43.17%)
- Extinction: 0.424602 (42.46%)
- Completion: 0.143733 (14.37%)

### Methanol

Predicted outcome: **Extinction**

- Extinction: 0.876155 (87.62%)
- Disruption: 0.119935 (11.99%)
- Completion: 0.003910 (0.39%)

## Score differences

| Outcome | Heptane | Methanol | Difference (percentage points) |
|---|---:|---:|---:|
| Completion | 14.3733% | 0.3910% | -13.98 |
| Disruption | 43.1665% | 11.9935% | -31.17 |
| Extinction | 42.4602% | 87.6155% | +45.16 |

## Interpretation

At the selected input conditions, the model produces different outcome scores for Heptane and Methanol. This is a model-behavior comparison, not causal evidence or an independently validated physical prediction.

## Limitations

- The model was fitted on all 273 eligible records.
- Model scores are not calibrated physical probabilities.
- This comparison uses one selected set of input conditions.
- The result does not establish a general causal fuel effect.
- The model is not suitable for operational fire-safety decisions.
