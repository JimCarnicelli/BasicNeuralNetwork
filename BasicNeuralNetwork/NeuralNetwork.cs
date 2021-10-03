using System;

namespace BasicNeuralNetwork {
    class NeuralNetwork {

        private static Random random = new Random();
        public static float NextRandom(float min, float max) {
            return (float)random.NextDouble() * (max - min) - min;
        }

        public int InputCount { get; private set; }
        public int OutputCount { get; private set; }
        public Layer[] Layers { get; private set; }
        public int LayerCount { get; private set; }
        public Layer InputLayer { get; private set; }
        public Layer OutputLayer { get; private set; }

        public NeuralNetwork() {
        }

        public Layer AddLayer(int neuronCount) {
            // Since we can't expand the array we'll construct a new one
            var newLayers = new Layer[LayerCount + 1];
            if (LayerCount > 0) Array.Copy(Layers, newLayers, LayerCount);

            // Interconnect layers
            Layer previousLayer = null;
            if (LayerCount > 0) previousLayer = newLayers[LayerCount - 1];

            // Construct the new layer
            var layer = new Layer(neuronCount, previousLayer);
            newLayers[LayerCount] = layer;

            // Interconnect layers
            if (LayerCount > 0) previousLayer.NextLayer = layer;

            // Cache some helpful properties
            if (LayerCount == 0) {
                InputLayer = layer;
                InputCount = neuronCount;
            }
            if (LayerCount == newLayers.Length - 1) {
                OutputLayer = layer;
                OutputCount = neuronCount;
            }

            // Emplace the new array and move on
            Layers = newLayers;
            LayerCount++;
            return layer;
        }

        /// <summary>
        /// Copy the array of input values to the input layer's .Output properties
        /// </summary>
        public void SetInputs(float[] inputs) {
            for (int n = 0; n < InputCount; n++) {
                InputLayer.Neurons[n].Output = inputs[n];
            }
        }

        /// <summary>
        /// Copy the output layer's .Output property values to the given array
        /// </summary>
        public void GetOutputs(float[] outputs) {
            for (int n = 0; n < OutputCount; n++) {
                outputs[n] = OutputLayer.Neurons[n].Output;
            }
        }

    }
}
