/**
 * https://jvcai.blogspot.com/2021/10/neural-network-in-c-with-multicore.html
 * 
 * MIT LICENSE
 * 
 * Copyright 2021 Jim Carnicelli
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this 
 * software and associated documentation files (the "Software"), to deal in the Software 
 * without restriction, including without limitation the rights to use, copy, modify, 
 * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit 
 * persons to whom the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or 
 * substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS IN THE SOFTWARE.
 * 
 */
namespace BasicNeuralNetwork {
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
