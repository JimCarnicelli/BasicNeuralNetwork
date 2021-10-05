using System;

namespace BasicNeuralNetwork {
    class Program {

        static void Main(string[] args) {

            var nn = new NeuralNetwork();
            nn.AddLayer(2, true, ActivationFunctionEnum.TanH);
            nn.AddLayer(2, true, ActivationFunctionEnum.TanH);
            nn.AddLayer(1, true, ActivationFunctionEnum.TanH);

            float[][] training = new float[][] {
                new float[] { 0, 0,   0 },
                new float[] { 0, 1,   1 },
                new float[] { 1, 0,   1 },
                new float[] { 1, 1,   0 },
            };

            int maxIterations = 10;
            int i = 0;
            while (i < maxIterations) {

                int trainingCase = NeuralNetwork.NextRandomInt(0, training.Length);
                var trainingData = training[trainingCase];
                nn.SetInputs(trainingData);
                nn.FeedForward();

                nn.TrainingOutputs[0] = trainingData[2];
                nn.Backpropagate();

                #region Output state

                const int colWidth = 9;
                string line = "";
                foreach (var neuron in nn.Layers[1].Neurons) {
                    line += " |";
                    //line += RightJustify(neuron.Output.ToString("0.000"), colWidth);
                    line += RightJustify(neuron.Bias.ToString("0.000"), colWidth);
                    foreach (var iw in neuron.InputWeights) {
                        line += RightJustify(iw.ToString("0.000"), colWidth);
                    }
                }
                line += "  |";
                foreach (var neuron in nn.Layers[2].Neurons) {
                    line += "|";
                    //line += RightJustify(neuron.Output.ToString("0.000"), colWidth);
                    line += RightJustify(neuron.Bias.ToString("0.000"), colWidth);
                    foreach (var iw in neuron.InputWeights) {
                        line += RightJustify(iw.ToString("0.000"), colWidth);
                    }
                }
                Console.WriteLine(
                    RightJustify("" + i, 8) +
                    line + " > " +
                    RightJustify("" + nn.Classify(), 2)
                );

                #endregion

                i++;
            }


            Console.WriteLine(nn.ToJson());
            Console.WriteLine("Done");
            Console.Beep();
            Console.ReadLine();
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
