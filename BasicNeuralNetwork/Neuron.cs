using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicNeuralNetwork {
    class Neuron {

        public Layer Layer;
        public float[] InputWeights;
        public float Bias;
        public float Output;
        public float Error;

        public Neuron(Layer layer) {
            Layer = layer;
            if (Layer.PreviousLayer != null) {
                InputWeights = new float[Layer.PreviousLayer.NeuronCount];
            }
        }

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
