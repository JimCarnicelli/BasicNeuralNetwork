using System;
using System.Collections.Generic;

namespace BasicNeuralNetwork {
    class Program {

        static void Main(string[] args) {

            RunXorDemo();
            //RunAsciiDemo();

            Console.WriteLine("Done");
            //Console.Beep();
            Console.ReadLine();
        }


        static void RunXorDemo() {
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

            int maxIterations = 30000;
            var corrects = new List<bool>();
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

                nn.Backpropagate();

                if (i > 0 && i % 1000 == 0) {
                    #region Output state

                    const int colWidth = 6;
                    string line = "";
                    foreach (var neuron in nn.Layers[1].Neurons) {
                        line += " |";
                        foreach (var iw in neuron.InputWeights) {
                            line += RightJustify(iw.ToString("0.00"), colWidth);
                        }
                        line += RightJustify(neuron.Bias.ToString("0.00"), colWidth);
                        line += RightJustify(neuron.Output.ToString("0.00"), colWidth);
                    }
                    line += " |";
                    foreach (var neuron in nn.Layers[2].Neurons) {
                        line += "|";
                        foreach (var iw in neuron.InputWeights) {
                            line += RightJustify(iw.ToString("0.00"), colWidth);
                        }
                        line += RightJustify(neuron.Bias.ToString("0.00"), colWidth);
                        line += RightJustify(neuron.Output.ToString("0.00"), colWidth);
                    }

                    Console.WriteLine(
                        RightJustify("" + i, 7) +
                        line + "  >" +
                        RightJustify("" + nn.InputLayer.Neurons[0].Output, 2) +
                        RightJustify("" + nn.InputLayer.Neurons[1].Output, 2) +
                        " : " +
                        (isCorrect ? "Y" : "N") +
                        RightJustify((percentCorrect * 100).ToString("0.0") + "%", 8)
                    );

                    #endregion
                }

                i++;
            }
        }


        static void RunAsciiDemo() {

        }


        static string LeftJustify(string text, int width) {
            while (text.Length < width) text = text + " ";
            return text;
        }
        static string RightJustify(string text, int width) {
            while (text.Length < width) text = " " + text;
            return text;
        }

    }
}
