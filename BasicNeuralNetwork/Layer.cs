using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicNeuralNetwork {
    class Layer {

        public Neuron[] Neurons;
        public Layer PreviousLayer;
        public Layer NextLayer;
        public int NeuronCount { get; private set; }

        public Layer(int neuronCount, Layer previousLayer) {
            // Put weights in the range of -0.5 to 0.5
            const float randomWeightRadius = 0.5f;

            PreviousLayer = previousLayer;
            NeuronCount = neuronCount;
            Neurons = new Neuron[NeuronCount];
            for (int n = 0; n < NeuronCount; n++) {
                Neuron neuron = new Neuron(this);
                neuron.Randomize(randomWeightRadius);
                Neurons[n] = neuron;
            }
        }

    }
}
