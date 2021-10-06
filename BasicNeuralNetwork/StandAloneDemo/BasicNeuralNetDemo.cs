using System;
using System.Collections.Generic;

namespace BasicNeuralNetworkDemo {

    class Program {

        static void DemoMain(string[] args) {

            var nn = new NeuralNetwork();
            nn.AddLayer(2);
            nn.AddLayer(2, true, ActivationFunctionEnum.TanH, 0.01f);
            nn.AddLayer(1, true, ActivationFunctionEnum.TanH, 0.01f);

            float[][] training = new float[][] {
                new float[] { 0, 0,   0 },
                new float[] { 0, 1,   1 },
                new float[] { 1, 0,   1 },
                new float[] { 1, 1,   0 },
            };

            Console.WriteLine("Iteration     Inputs    Output   Valid?      Accuracy");

            int maxIterations = 1000000;
            var corrects = new List<bool>();
            int flawlessRuns = 0;
            int i = 0;
            while (i < maxIterations) {

                int trainingCase = NeuralNetwork.NextRandomInt(0, training.Length);
                var trainingData = training[trainingCase];
                nn.SetInputs(trainingData);
                nn.FeedForward();

                nn.TrainingOutputs[0] = trainingData[2];

                bool isCorrect = (nn.OutputLayer.Neurons[0].Output < 0.5 ? 0 : 1) == nn.TrainingOutputs[0];
                corrects.Add(isCorrect);
                while (corrects.Count > 100) corrects.RemoveAt(0);
                float percentCorrect = 0;
                foreach (var correct in corrects) if (correct) percentCorrect += 1;
                percentCorrect /= corrects.Count;

                if (percentCorrect == 1) flawlessRuns++;
                else flawlessRuns = 0;

                nn.Backpropagate();

                if (i % 100 == 0) {
                    #region Output state

                    Console.WriteLine(
                        RightJustify(i.ToString("#,##0"), 9) + "    " +
                        trainingData[0] +
                        " xor " +
                        trainingData[1] + " = " +
                        RightJustify("" + nn.OutputLayer.Neurons[0].Output.ToString("0.000"), 7) + "  " +
                        (isCorrect ? "       " : "(wrong)") +
                        RightJustify((percentCorrect * 100).ToString("0.0") + "% ", 12) +
                        RenderPercent(percentCorrect * 100)
                    );

                    #endregion
                }

                if (flawlessRuns == 1000) {
                    Console.WriteLine("I've had " + flawlessRuns.ToString("#,##0") + " flawless predictions recently. Continue anyway?");
                    Console.Beep();
                    Console.ReadLine();
                }

                i++;
            }

            Console.WriteLine("Done");
            Console.Beep();
            Console.ReadLine();
        }

        static string RenderPercent(float percent) {
            float value = percent / 10f;
            if (value < 0.5) return "|          |";
            if (value < 1.5) return "|-         |";
            if (value < 2.5) return "|--        |";
            if (value < 3.5) return "|---       |";
            if (value < 4.5) return "|----      |";
            if (value < 5.5) return "|-----     |";
            if (value < 6.5) return "|------    |";
            if (value < 7.5) return "|-------   |";
            if (value < 8.5) return "|--------  |";
            if (value < 9.5) return "|--------- |";
            return "|----------|";
        }

        static string RightJustify(string text, int width) {
            while (text.Length < width) text = " " + text;
            return text;
        }

    }


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

    public class NeuralNetwork {

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
        public Layer AddLayer(
            int neuronCount,
            bool randomize = false,
            ActivationFunctionEnum activationFunction = ActivationFunctionEnum.TanH,
            float learningRate = 0.01f
        ) {
            // Since we can't expand the array we'll construct a new one
            var newLayers = new Layer[LayerCount + 1];
            if (LayerCount > 0) Array.Copy(Layers, newLayers, LayerCount);

            // Interconnect layers
            Layer previousLayer = null;
            if (LayerCount > 0) previousLayer = newLayers[LayerCount - 1];

            // Construct the new layer
            var layer = new Layer(neuronCount, previousLayer);
            layer.ActivationFunction = activationFunction;
            layer.LearningRate = learningRate;
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
            for (int l = 1; l < LayerCount; l++) {
                var layer = Layers[l];
                layer.FeedForward();
            }
        }

        /// <summary>
        /// One iteration of backpropagation training using inputs and training outputs after .Predict() was called on the same
        /// </summary>
        public void Backpropagate() {
            for (int l = LayerCount - 1; l > 0; l--) {
                var layer = Layers[l];
                layer.Backpropagate(TrainingOutputs);
            }
        }

        /// <summary>
        /// Returns a random float in the range from min to max (inclusive)
        /// </summary>
        public static float NextRandom(float min, float max) {
            return (float)random.NextDouble() * (max - min) + min;
        }
        /// <summary>
        /// Returns a random int that is at least min and less than max
        /// </summary>
        public static int NextRandomInt(int min, int max) {
            return random.Next(min, max);
        }
        private static Random random = new Random();

    }


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

    }


    public class Neuron {

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
            if (layer.PreviousLayer != null) {
                InputWeights = new float[layer.PreviousLayer.NeuronCount];
            }
        }

        /// <summary>
        /// Forget all prior training by randomizing my input weights and bias
        /// </summary>
        public void Randomize(float radius) {
            if (InputWeights != null) {
                for (int i = 0; i < InputWeights.Length; i++) {
                    InputWeights[i] = NeuralNetwork.NextRandom(-radius, radius);
                }
            }
            Bias = NeuralNetwork.NextRandom(-radius, radius);
        }

    }

}
