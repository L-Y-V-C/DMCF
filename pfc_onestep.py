#!/usr/bin/env python3
"""(PFC) Predicción a 1 paso ("teacher forcing") con DMCF.

Igual que `run_pipeline.py --split test`, pero en cada frame t el modelo recibe el estado REAL (GT) del
frame t y predice t+1; no se realimenta con sus propias predicciones. Sirve para visualizar el error local
del modelo; NO es una simulación (para eso está el rollout normal). No modifica el código del repo:
solo reemplaza Simulator.run_rollout en tiempo de ejecución.

Uso (mismos argumentos que run_pipeline.py):
  python pfc_onestep.py --cfg_file configs/Liquid3d.yml --dataset_path DATA --ckpt_path CKPT --split test \
      --pipeline.data_generator.test.time_end 61 --pipeline.data_generator.test.cnt 2 --pipeline.output_dir OUT
"""
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import run_pipeline
from pipelines.simulator import Simulator


def run_rollout_one_step(self, inputs, timesteps=2):
    def state(data, t):
        return [tf.convert_to_tensor(data['pos'][t]), tf.convert_to_tensor(data['vel'][t]),
                tf.convert_to_tensor(data['grav'][t]) if data['grav'][0] is not None else None, None,
                tf.convert_to_tensor(data['box'][0]), tf.convert_to_tensor(data['box_normals'][0])]

    results = [[state(d, 0)] for d in inputs]
    self.run_inference([state(inputs[0], 0)])  # inicialización (igual que el original)
    for t in tqdm(range(timesteps - 1), 'one-step'):
        for i, d in enumerate(inputs):
            results[i].append(self.run_inference([state(d, t)])[0])
    return results


Simulator.run_rollout = run_rollout_one_step

if __name__ == '__main__':
    run_pipeline.main()
