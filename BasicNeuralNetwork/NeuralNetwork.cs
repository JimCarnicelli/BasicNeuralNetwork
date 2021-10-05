using System;
using System.Text.Json;

namespace BasicNeuralNetwork {

    public enum ActivationFunctionEnum {
        /// <summary> Rectified Linear Unit </summary>
        ReLU,
        /// <summary> Leaky Rectified Linear Unit </summary>
        LReLU,
        /// <summary> Logistic sigmoid </summary>
        Sigmoid,
        /// <summary> Hyperbolic tangent </summary>
        TanH,
        /// <summary> Softmax function </summary>
        Softmax,
    }

    class NeuralNetwork {

        /// <summary>
        /// The layers of neurons from input (0) to output (N)
        /// </summary>
        public Layer[] Layers { get; private set; }

        /// <summary>
        /// Equivalent to Layers.Length
        /// </summary>
        public int LayerCount { get; private set; }

        /// <summary>
        /// Equivalent to InputLayer.NeuronCount
        /// </summary>
        public int InputCount { get; private set; }

        /// <summary>
        /// Equivalent to OutputLayer.NeuronCount
        /// </summary>
        public int OutputCount { get; private set; }

        /// <summary>
        /// Equivalent to Layers[0]
        /// </summary>
        public Layer InputLayer { get; private set; }

        /// <summary>
        /// Equivalent to Layers[LayerCount - 1]
        /// </summary>
        public Layer OutputLayer { get; private set; }

        /// <summary>
        /// Provides the desired output values for use in backpropagation training
        /// </summary>
        public float[] TrainingOutputs { get; private set; }

        public NeuralNetwork() { }

        /// <summary>
        /// Constructs and adds a new neuron layer to .Layers
        /// </summary>
        public Layer AddLayer(int neuronCount, bool randomize, ActivationFunctionEnum activationFunction = ActivationFunctionEnum.TanH) {
            // Since we can't expand the array we'll construct a new one
            var newLayers = new Layer[LayerCount + 1];
            if (LayerCount > 0) Array.Copy(Layers, newLayers, LayerCount);

            // Interconnect layers
            Layer previousLayer = null;
            if (LayerCount > 0) previousLayer = newLayers[LayerCount - 1];

            // Construct the new layer
            var layer = new Layer(neuronCount, previousLayer);
            if (randomize) layer.Randomize();
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
                TrainingOutputs = new float[neuronCount];
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

        /// <summary>
        /// Interpret the output array as a singular category (0, 1, 2, ...) or -1 (none)
        /// </summary>
        public int Classify() {
            float maxValue = 0;
            int bestIndex = -1;
            for (int o = 0; o < OutputCount; o++) {
                float value = OutputLayer.Neurons[o].Output;
                if (value > maxValue) {
                    bestIndex = o;
                    maxValue = value;
                }
            }
            if (maxValue == 0) return -1;
            return bestIndex;
        }

        /// <summary>
        /// Copy the given array's values to the .TrainingOutputs property
        /// </summary>
        public void SetTrainingOutputs(float[] outputs) {
            Array.Copy(outputs, TrainingOutputs, OutputCount);
        }

        /// <summary>
        /// Flipside of .Classify() that sets .TrainingOutputs to all zeros and the given index to one
        /// </summary>
        public void SetTrainingClassification(int value) {
            for (int o = 0; o < OutputCount; o++) {
                if (o == value) {
                    TrainingOutputs[o] = 1;
                } else {
                    TrainingOutputs[o] = 0;
                }
            }
        }

        /// <summary>
        /// Feed .Inputs forward to populate .Outputs
        /// </summary>
        public void FeedForward() {
            // TODO: Implement
            for (int l = 1; l < LayerCount; l++) {
                var layer = Layers[l];
                layer.FeedForward();
            }
        }

        /// <summary>
        /// One iteration of backpropagation training using inputs and training outputs after .Predict() was called on the same
        /// </summary>
        public void Backpropagate() {
            // TODO: Implement
        }

        /// <summary>
        /// Convert my internal state into a JSON representation
        /// </summary>
        public string ToJson(string indent = "") {
            string json = "{\n";
            json += indent + "  \"layers\": [\n";
            json += indent + "    {\n";
            json += indent + "      \"neuronCount\": " + InputCount + "\n";
            json += indent + "    }";
            for (int l = 1; l < LayerCount; l++) {
                json += ", ";
                json += Layers[l].ToJson(indent + "    ");
            }
            json += "\n";
            json += indent + "  ]\n";
            json += indent + "}";
            return json;
        }

        /// <summary>
        /// Constructs my internal state from the given JSON text
        /// </summary>
        public void FromJson(string json) {
            using JsonDocument doc = JsonDocument.Parse(json);
            FromJson(doc.RootElement);
        }

        /// <summary>
        /// Constructs my internal state from the given element as part of a larger JSON document
        /// </summary>
        public void FromJson(JsonElement networkElem) {
            var layerElems = networkElem.GetProperty("layers");
            var layerCount = layerElems.GetArrayLength();
            Layers = new Layer[0];

            // First layer is just the simple dummy input neurons
            var neuronCount = layerElems[0].GetProperty("neuronCount").GetInt32();
            AddLayer(neuronCount, false);

            for (int l = 1; l < layerCount; l++) {
                var layerElem = layerElems[l];
                neuronCount = layerElem.GetProperty("neuronCount").GetInt32();
                var layer = AddLayer(neuronCount, false);
                var inputCount = Layers[LayerCount - 2].NeuronCount;
                layer.ActivationFunction = Enum.Parse<ActivationFunctionEnum>(layerElem.GetProperty("activationFunction").GetString());
                var inputWeights = layerElem.GetProperty("inputWeights").GetString().Split(",");
                var biasWeights = layerElem.GetProperty("biasWeights").GetString().Split(",");
                for (int n = 0; n < neuronCount; n++) {
                    var neuron = layer.Neurons[n];
                    for (int i = 0; i < inputCount; i++) {
                        neuron.InputWeights[i] = float.Parse(inputWeights[n * inputCount + i]);
                    }
                    neuron.Bias = float.Parse(biasWeights[n]);
                }
            }
        }

        /// <summary>
        /// Returns a random float in the range from min to max (inclusive)
        /// </summary>
        public static float NextRandom(float min, float max) {
            return (float)random.NextDouble() * (max - min) + min;
        }
        public static int NextRandomInt(int min, int max) {
            return random.Next(min, max);
        }
        private static Random random = new Random();

    }
}
