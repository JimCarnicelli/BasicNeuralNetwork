﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicNeuralNetwork {
    class Neuron {

        /// <summary>
        /// The layer I belong to
        /// </summary>
        public Layer Layer;

        /// <summary>
        /// The weight I put on each of my inputs when computing my output as my essential learned memory
        /// </summary>
        public float[] InputWeights;

        /// <summary>
        /// My bias is also part of my learned memory
        /// </summary>
        public float Bias;

        /// <summary>
        /// My feed-forward computed output
        /// </summary>
        public float Output;

        /// <summary>
        /// My back-propagation computed error
        /// </summary>
        public float Error;

        public Neuron(Layer layer) {
            Layer = layer;
            if (Layer.PreviousLayer != null) {
                InputWeights = new float[Layer.PreviousLayer.NeuronCount];
            }
        }

        /// <summary>
        /// Forget all prior training by randomizing my input weights and bias
        /// </summary>
        public void Randomize(float radius) {
            if (InputWeights != null) {
                for (int i = 0; i < Layer.PreviousLayer.NeuronCount; i++) {
                    InputWeights[i] = NeuralNetwork.NextRandom(-radius, radius);
                }
            }
            Bias = NeuralNetwork.NextRandom(-radius, radius);
        }

    }
}
