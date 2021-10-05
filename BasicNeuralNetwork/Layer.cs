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

        public float LearningRate = 0.01f;

        public ActivationFunctionEnum ActivationFunction = ActivationFunctionEnum.TanH;

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
            foreach (Neuron neuron in Neurons) {
                neuron.Randomize(randomWeightRadius);
            }
        }

        /// <summary>
        /// Feed-forward algorithm for this layer
        /// </summary>
        public void FeedForward() {
            // TODO: Implement
        }

        /// <summary>
        /// Feed-forward algorithm for this output layer
        /// </summary>
        public void BackpropagateOutput() {
            // TODO: Implement
        }

        /// <summary>
        /// Feed-forward algorithm for this hidden layer
        /// </summary>
        public void BackpropagateHidden() {
            // TODO: Implement
        }

        /// <summary>
        /// Convert my internal state into a JSON representation
        /// </summary>
        public string ToJson(string indent = "") {
            string jsonAccum = "";
            string json = "{\n";
            json += indent + "  \"neuronCount\": " + NeuronCount + ",\n";
            json += indent + "  \"learningRate\": " + LearningRate + ",\n";
            json += indent + "  \"activationFunction\": \"" + ActivationFunction + "\",\n";

            // Input weights for all neurons
            json += indent + "  \"inputWeights\": \"";
            bool first = true;
            foreach (Neuron neuron in Neurons) {
                foreach (float inputWeight in neuron.InputWeights) {
                    if (first) first = false;
                    else json += ",";
                    json += inputWeight.ToString("G9");
                    if (json.Length > 1024) {
                        jsonAccum += json;
                        json = "";
                    }
                }
            }
            json += "\",\n";

            // Biases for each neuron
            json += indent + "  \"biasWeights\": \"";
            first = true;
            foreach (Neuron neuron in Neurons) {
                if (first) first = false;
                else json += ",";
                json += neuron.Bias.ToString("G9");
                if (json.Length > 1024) {
                    jsonAccum += json;
                    json = "";
                }
            }
            json += "\"\n";

            json += indent + "}";
            return jsonAccum + json;
        }

    }
}
