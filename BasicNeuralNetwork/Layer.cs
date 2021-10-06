using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicNeuralNetwork {
    public class Layer {

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
        /// A tunable parameter that trades shorter training times for greater final accuracy
        /// </summary>
        public float LearningRate = 0.01f;

        /// <summary>
        /// How to transform the summed-up scalar output value of each neuron during feed forward
        /// </summary>
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
            foreach (var neuron in Neurons) {

                // Sum up the previous layer's outputs multiplied by this neuron's weights for each
                float sigma = 0;
                for (int i = 0; i < PreviousLayer.NeuronCount; i++) {
                    sigma += PreviousLayer.Neurons[i].Output * neuron.InputWeights[i];
                }
                sigma += neuron.Bias;  // Add in each neuron's bias too

                // Shape the output using the activation function
                float output = ActivationFn(sigma);
                neuron.Output = output;
            }

            // The Softmax activation function requires extra processing of aggregates
            if (ActivationFunction == ActivationFunctionEnum.Softmax) {
                // Find the max output value
                float max = float.NegativeInfinity;
                foreach (var neuron in Neurons) {
                    if (neuron.Output > max) max = neuron.Output;
                }
                // Compute the scale
                float scale = 0;
                foreach (var neuron in Neurons) {
                    scale += (float)Math.Exp(neuron.Output - max);
                }
                // Shift and scale the outputs
                foreach (var neuron in Neurons) {
                    neuron.Output = (float)Math.Exp(neuron.Output - max) / scale;
                }
            }
        }

        /// <summary>
        /// Backpropagation algorithm
        /// </summary>
        public void Backpropagate(float[] trainingOutputs) {

            // Compute error for each neuron
            for (int n = 0; n < NeuronCount; n++) {
                var neuron = Neurons[n];
                float output = neuron.Output;

                if (NextLayer == null) {  // Output layer
                    var error = trainingOutputs[n] - output;
                    neuron.Error = error * ActivationFnDerivative(output);
                } else {  // Hidden layer
                    float error = 0;
                    for (int o = 0; o < NextLayer.NeuronCount; o++) {
                        var nextNeuron = NextLayer.Neurons[o];
                        var iw = nextNeuron.InputWeights[n];
                        error += nextNeuron.Error * iw;

                        // TODO: Implement input weight regularization
                        /*
                        // Calculate L1 (lasso) for regularization
                        l1Penalty += (iw < 0 ? -iw : iw);  // Absolute value of the weight

                        // Calculate L2 (weight decay) for regularization
                        l1Penalty += iw * iw;  // Weight squared
                        */

                    }
                    neuron.Error = error * ActivationFnDerivative(output);
                }
            }

            // Adjust weights of each neuron
            for (int n = 0; n < NeuronCount; n++) {
                var neuron = Neurons[n];

                // Update this neuron's bias
                var gradient = neuron.Error;
                neuron.Bias += gradient * LearningRate;

                // Update this neuron's input weights
                for (int i = 0; i < PreviousLayer.NeuronCount; i++) {
                    gradient = neuron.Error * PreviousLayer.Neurons[i].Output;
                    neuron.InputWeights[i] += gradient * LearningRate;
                }
            }

        }

        private float ActivationFn(float value) {
            switch (ActivationFunction) {
                case ActivationFunctionEnum.ReLU:
                    if (value < 0) return 0;
                    return value;
                case ActivationFunctionEnum.LReLU:
                    if (value < 0) return value * 0.01f;
                    return value;
                case ActivationFunctionEnum.Sigmoid:
                    return (float)(1 / (1 + Math.Exp(-value)));
                case ActivationFunctionEnum.TanH:
                    return (float)Math.Tanh(value);
                case ActivationFunctionEnum.Softmax:
                    return value;  // This is only the first part of summing up all the values
            }
            return value;
        }

        private float ActivationFnDerivative(float value) {
            switch (ActivationFunction) {
                case ActivationFunctionEnum.ReLU:
                    if (value > 0) return 1;
                    return 0;
                case ActivationFunctionEnum.LReLU:
                    if (value > 0) return 1;
                    return 0.01f;
                case ActivationFunctionEnum.Sigmoid:
                    return value * (1 - value);
                case ActivationFunctionEnum.TanH:
                    return 1 - value * value;
                case ActivationFunctionEnum.Softmax:
                    return (1 - value) * value;
            }
            return 0;
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
