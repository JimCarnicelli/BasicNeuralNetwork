using System;
using System.Text.Json;

namespace BasicNeuralNetworkV1
{

    // https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/
    public enum ActivationFunctionEnum {
        /// <summary> Rectified Linear Unit </summary>
        ReLU,
        /// <summary> Leaky Rectified Linear Unit </summary>
        LReLU,
        /// <summary> Logistic sigmoid </summary>
        Sigmoid,
        /// <summary> Hyperbolic tangent </summary>
        TanH,
    }

    public class NeuralNetworkV1
    {
        public Random random = new Random();
        public AnnLayerBase[] layers;
        public float[] trainingOutputs;

        public float[] inputs { get {
            return layers[0].outputs;
        }}

        public float[] outputs { get {
            return layers[layers.Length - 1].outputs;
        }}

        public NeuralNetworkV1(
            int[] layerNeuronCounts,
            ActivationFunctionEnum activationFunction = ActivationFunctionEnum.TanH,
            bool withTraining = false
        ) {
            layers = new AnnLayerBase[layerNeuronCounts.Length];
            // First layer is the dummy layer for inputs
            layers[0] = new AnnLayerBase(layerNeuronCounts[0]);
            // All other layers contain active perceptron neurons
            for (int i = 1; i < layerNeuronCounts.Length; i++)
            {
                layers[i] = new AnnLayer(
                    this,
                    layers[i - 1],  // Previous layer
                    layerNeuronCounts[i],  // Neuron count
                    activationFunction,
                    withTraining  // Randomize
                );
                // Connect the previous layer forward to this one now that it's constructed
                if (i > 1) ((AnnLayer)layers[i - 1]).nextLayer = (AnnLayer)layers[i];
            }

            // Don't bother allocating memory for training if we're not going to do any
            if (withTraining)
            {
                trainingOutputs = new float[outputs.Length];
            }
        }

        public NeuralNetworkV1(string json, bool withTraining = false) {
            FromJson(json);
            if (withTraining) {
                trainingOutputs = new float[outputs.Length];
            }
        }

        public NeuralNetworkV1 Clone() {
            var json = ToJson();
            var nn = new NeuralNetworkV1(json);
            return nn;
        }

        public void SetInputs(float[] inputs) {
            for (int i = 0; i < inputs.Length; i++) {
                this.inputs[i] = inputs[i];
            }
        }

        public void SetTrainingOutputs(float[] outputs) {
            for (int i = 0; i < outputs.Length; i++) {
                trainingOutputs[i] = outputs[i];
            }
        }

        /// <summary>
        /// Translate the classification (0, 1, 2, ...) to an output string (eg [0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        /// </summary>
        public void SetTrainingOutputs(int classification) {
            for (int i = 0; i < outputs.Length; i++) {
                if (i == classification) {
                    trainingOutputs[i] = 1;
                } else {
                    trainingOutputs[i] = 0;
                }
            }
        }

        /// <summary>
        /// Interpret the output array as a singular category (0, 1, 2, ...) or -1 (none)
        /// </summary>
        public int Classify() {
            float maxValue = 0;
            int bestIndex = -1;
            for (int i = 0; i < outputs.Length; i++) {
                float value = outputs[i];
                if (value > maxValue) {
                    bestIndex = i;
                    maxValue = value;
                }
            }
            if (maxValue == 0) return -1;
            return bestIndex;
        }

        /// <summary>
        /// Behaving
        /// </summary>
        public void FeedForward()
        {
            for (int i = 1; i < layers.Length; i++)
            {
                ((AnnLayer)layers[i]).FeedForward();
            }
        }

        /// <summary>
        /// Learning
        /// </summary>
        public void FeedBack()
        {
            if (trainingOutputs == null) throw new Exception("Not constructed for training");
            ((AnnLayer)layers[layers.Length - 1]).FeedBackOutput(trainingOutputs);
            for (int i = layers.Length - 2; i >= 1; i--)
            {
                ((AnnLayer)layers[i]).FeedBackHidden();
            }
        }

        public int CountInputWeights() {
            int count = 0;
            for (int i = 1; i < layers.Length; i++) {
                var layer = (AnnLayer)layers[i];
                count += layer.inputWeights.Length;
            }
            return count;
        }

        /// <summary>
        /// Calculate the sum of L1 values for all non-input layers
        /// </summary>
        public float CalculateWeightL1() {
            float sum = 0;
            for (int i = 1; i < layers.Length; i++) {
                var layer = (AnnLayer)layers[i];
                sum += layer.CalculateWeightL1();
            }
            return sum;
        }

        /// <summary>
        /// Calculate the sum of L2 values for all non-input layers
        /// </summary>
        public float CalculateWeightL2() {
            float sum = 0;
            for (int i = 1; i < layers.Length; i++) {
                var layer = (AnnLayer)layers[i];
                sum += layer.CalculateWeightL2();
            }
            return sum;
        }

        /// <summary>
        /// Convenience function for setting the learning rate setting on all layers to the same value
        /// </summary>
        public void SetAllLearningRates(float value) {
            for (int i = 1; i < layers.Length; i++) {
                ((AnnLayer)layers[i]).learningRate = value;
            }
        }

        public void SetAllLambdas(float l1Lambda, float l2Lambda) {
            for (int i = 1; i < layers.Length; i++) {
                ((AnnLayer)layers[i]).l1Lambda = l1Lambda;
                ((AnnLayer)layers[i]).l2Lambda = l2Lambda;
            }
        }

        /// <summary>
        /// Convert my internal state into a JSON representation
        /// </summary>
        public string ToJson(string indent = "") {
            string json = indent + "{\n";
            json += indent + "  \"layers\": [\n";
            json += indent + "    {\n";
            json += indent + "      \"neuronCount\": " + inputs.Length + "\n";
            json += indent + "    },\n";
            for (int i = 1; i < layers.Length; i++) {
                json += ((AnnLayer)layers[i]).ToJson(indent + "    ");
                if (i < layers.Length - 1) json += ",";
                json += "\n";
            }
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
        public void FromJson(JsonElement elem) {
            var layerList = elem.GetProperty("layers");
            var layerCount = layerList.GetArrayLength();
            layers = new AnnLayerBase[layerCount];

            // First layer is just the simple dummy input neurons
            var neuronCount = layerList[0].GetProperty("neuronCount").GetInt32();
            layers[0] = new AnnLayerBase(neuronCount);

            for (int i = 1; i < layerCount; i++) {
                layers[i] = new AnnLayer(this, layerList[i], layers[i - 1]);
                // Connect the previous layer forward to this one now that it's constructed
                if (i > 1) ((AnnLayer)layers[i - 1]).nextLayer = (AnnLayer)layers[i];
            }
        }

        /// <summary>
        /// Convert the given single floating point value to a hexadecimal string representation for storage
        /// </summary>
        public string FloatToHex(float f) {
            var bytes = BitConverter.GetBytes(f);
            var i = BitConverter.ToInt32(bytes, 0);
            return i.ToString("X8");
        }

        /// <summary>
        /// Convert the hexadecimal string representation back into a single floating point value
        /// </summary>
        public float HexToFloat(string s) {
            var i = Convert.ToInt32("0x" + s, 16);
            var bytes = BitConverter.GetBytes(i);
            return BitConverter.ToSingle(bytes, 0);
        }

    }

    /// <summary>
    /// A hidden or output layer (but not an input layer)
    /// </summary>
    public class AnnLayer : AnnLayerBase
    {
        public NeuralNetworkV1 network;  // The parent ANN
        public AnnLayerBase previousLayer;  // The previous layer gives me my inputs
        public AnnLayer nextLayer;
        public float[] inputWeights;  // count = inputs * neurons
        public float[] biasWeights;  // count = neurons
        public float[] errors;  // count = neurons
        public float learningRate = 0.01f;
        public float l1Lambda = 0.01f;
        public float l2Lambda = 0.01f;
        public ActivationFunctionEnum activationFunction;

        private int neuronCount;
        private int inputCount;  // A copy of previousLayer.outputs.Length

        public AnnLayer(
            NeuralNetworkV1 network,
            AnnLayerBase previousLayer,
            int neuronCount,
            ActivationFunctionEnum activationFunction,
            bool randomize
        ) : base(neuronCount) {
            this.network = network;
            this.previousLayer = previousLayer;
            this.neuronCount = neuronCount;
            inputCount = previousLayer.outputs.Length;
            inputWeights = new float[neuronCount * inputCount];
            biasWeights = new float[neuronCount];
            errors = new float[neuronCount];
            this.activationFunction = activationFunction;
            if (randomize) Randomize();
        }

        public AnnLayer(NeuralNetworkV1 nn, JsonElement elem, AnnLayerBase previousLayer) :
            this(
                nn,
                previousLayer,
                elem.GetProperty("neuronCount").GetInt32(),
                Enum.Parse<ActivationFunctionEnum>(elem.GetProperty("activationFunction").GetString()),
                false  // Randomize
            )
        {
            // Input weights
            var weights = elem.GetProperty("inputWeights").GetString().Split(",");
            for (int i = 0; i < weights.Length; i++) {
                inputWeights[i] = nn.HexToFloat(weights[i]);
            }
            // Bias weights
            weights = elem.GetProperty("biasWeights").GetString().Split(",");
            for (int i = 0; i < weights.Length; i++) {
                biasWeights[i] = nn.HexToFloat(weights[i]);
            }
        }

        public void Randomize() {
            const float scale = 2f;
            // Randomize weights
            for (int n = 0; n < neuronCount; n++) {
                for (int i = 0; i < inputCount; i++) {
                    inputWeights[n * inputCount + i] = (float)(scale * (network.random.NextDouble() - 0.5f));
                }
                biasWeights[n] = (float)(scale * (network.random.NextDouble() - 0.5f));
            }
        }

        public void FeedForward() {
            for (int n = 0; n < neuronCount; n++) {
                float sigma = 0;
                for (int i = 0; i < inputCount; i++) {
                    sigma += (previousLayer.outputs[i] * inputWeights[n * inputCount + i]);
                }
                sigma += biasWeights[n];
                outputs[n] = ActivationFunction(sigma);
            }
        }

        /// <summary>
        /// The back-propagation algorithm specifically for the final, output layer
        /// </summary>
        public void FeedBackOutput(float[] targetOutputs) {
            for (int n = 0; n < neuronCount; n++) {
                // Compute the error
                float output = outputs[n];
                float error = ActivationDerivative(output) * (targetOutputs[n] - output);
                errors[n] = error;
                AdjustWeights(n, error, 0, 0);
            }
        }

        /// <summary>
        /// The back-propagation algorithm specifically for all hidden layers
        /// </summary>
        public void FeedBackHidden() {
            float l1Penalty = 0;
            float l2Penalty = 0;

            for (int n = 0; n < neuronCount; n++) {

                // Compute error for neuron
                float output = outputs[n];
                float error = 0;
                for (int o = 0; o < nextLayer.outputs.Length; o++) {
                    var iw = nextLayer.inputWeights[o * neuronCount + n];
                    error += (nextLayer.errors[o] * iw);

                    // Calculate L1 (lasso) for regularization
                    l1Penalty += (iw < 0 ? -iw : iw);  // Absolute value of the weight

                    // Calculate L2 (weight decay) for regularization
                    l1Penalty += iw * iw;  // Weight squared
                }
                error *= ActivationDerivative(output);
                errors[n] = error;
            }

            // Adjust weights of each neuron
            for (int n = 0; n < neuronCount; n++) {
                AdjustWeights(n, errors[n], l1Penalty, l2Penalty);
            }

        }

        /// <summary>
        /// Calculate the sum of absolute values of all input weights (L1)
        /// </summary>
        public float CalculateWeightL1() {
            float sum = 0;
            for (int j = 0; j < inputWeights.Length; j++) {
                var w = inputWeights[j];
                if (w < 0) w = -w;  // Absolute value
                sum += w;
            }
            return sum;
        }

        /// <summary>
        /// Calculate the sum of squares of all input weights (L2)
        /// </summary>
        public float CalculateWeightL2() {
            float sum = 0;
            for (int j = 0; j < inputWeights.Length; j++) {
                var w = inputWeights[j];
                sum += w * w;
            }
            return sum;
        }

        private void AdjustWeights(int n, float error, float l1Penalty, float l2Penalty) {

            // TODO: Figure out how to properly code L1 and L2 regularization
            if (error < 0) {
                l1Penalty = -l1Penalty;
                l2Penalty = -l2Penalty;
            }
            error += l1Lambda * l1Penalty;
            error += l2Lambda * l2Penalty;

            for (int w = 0; w < inputCount; w++) {
                inputWeights[n * inputCount + w] += error * previousLayer.outputs[w] * learningRate;
            }
            biasWeights[n] += error * learningRate;
        }

        private float ActivationFunction(float value) {
            switch (activationFunction) {
                case ActivationFunctionEnum.ReLU:
                    if (value > 0) return value;
                    return 0;
                case ActivationFunctionEnum.LReLU:
                    if (value > 0) return value;
                    return value * 0.1f;
                case ActivationFunctionEnum.Sigmoid:
                    if (value < -45.0) return 0;
                    else if (value > 45.0) return 1;
                    return (float)(1 / (1 + Math.Exp(-value)));
                case ActivationFunctionEnum.TanH:
                    if (value < -45.0) return -1;
                    else if (value > 45.0) return 1;
                    return (float)Math.Tanh(value);
            }
            return 0;
        }

        private float ActivationDerivative(float value) {
            switch (activationFunction) {
                case ActivationFunctionEnum.ReLU:
                    if (value > 0) return 1;
                    return 0;
                case ActivationFunctionEnum.LReLU:
                    if (value > 0) return 1;
                    return 0.1f;
                case ActivationFunctionEnum.Sigmoid:
                    {
                        var f = ActivationFunction(value);
                        return f * (1 - f);
                    }
                case ActivationFunctionEnum.TanH:
                    {
                        var f = ActivationFunction(value);
                        return 1 - f * f;
                    }
            }
            return 0;
        }

        /// <summary>
        /// Convert my internal state into a JSON representation
        /// </summary>
        public string ToJson(string indent = "") {
            string jsonAccum = "";
            string json = indent + "{\n";
            json += indent + "  \"neuronCount\": " + neuronCount + ",\n";
            json += indent + "  \"learningRate\": " + learningRate + ",\n";
            json += indent + "  \"activationFunction\": \"" + activationFunction + "\",\n";

            json += indent + "  \"inputWeights\": \"";
            for (int i = 0; i < inputWeights.Length; i++) {
                if (i > 0) json += ",";
                json += network.FloatToHex(inputWeights[i]);

                if (json.Length > 1024) {
                    jsonAccum += json;
                    json = "";
                }
            }
            json += "\",\n";

            json += indent + "  \"biasWeights\": \"";
            for (int i = 0; i < biasWeights.Length; i++) {
                if (i > 0) json += ",";
                json += network.FloatToHex(biasWeights[i]);

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

    /// <summary>
    /// Holds only the things needed by the input layer and acts as the basis for all others
    /// </summary>
    public class AnnLayerBase
    {
        public float[] outputs;

        public AnnLayerBase(int neuronCount)
        {
            outputs = new float[neuronCount];
        }

        public string outputsToString()
        {
            string text = "";
            for (int i = 0; i < outputs.Length; i++)
            {
                if (text != "") text += " | ";
                text += outputs[i].ToString("0.000");
            }
            return text;
        }

    }

}
