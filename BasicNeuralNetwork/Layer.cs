using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicNeuralNetwork {
    class Layer {

        /// <summary>
        /// All the neurons in this layer
        /// </summary>
        public Neuron[] Neurons;

        /// <summary>
        /// Reference to the earlier layer that I get my input from
        /// </summary>
        public Layer PreviousLayer;

        /// <summary>
        /// Reference to the later layer that gets its input from me
        /// </summary>
        public Layer NextLayer;

        /// <summary>
        /// Equivalent to Neurons.Length
        /// </summary>
        public int NeuronCount { get; private set; }

        public Layer(int neuronCount, Layer previousLayer) {
            PreviousLayer = previousLayer;
            NeuronCount = neuronCount;
            Neurons = new Neuron[NeuronCount];
            for (int n = 0; n < NeuronCount; n++) {
                Neuron neuron = new Neuron(this);
                Neurons[n] = neuron;
            }
        }

        /// <summary>
        /// Forget all prior training by randomizing all input weights and biases
        /// </summary>
        public void Randomize() {
            // Put weights in the range of -0.5 to 0.5
            const float randomWeightRadius = 0.5f;
            for (int n = 0; n < NeuronCount; n++) {
                Neurons[n].Randomize(randomWeightRadius);
            }
        }

    }
}
