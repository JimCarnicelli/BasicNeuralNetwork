using System;
using System.Collections.Generic;

namespace BasicNeuralNetwork {
    class Program {

        static void Main(string[] args) {

            //RunXorDemo();
            RunAsciiDemo();

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
            NeuralNetwork nn;

            /*
            const bool loadFromFile = false;
            var filePath = DataDirectory() + "/Character Neural Network.json";

            if (loadFromFile && File.Exists(filePath)) {
                var json = File.ReadAllText(filePath);
                nn = new NeuralNetworkV1(json, true);

            } else {
                nn = NewCharacterNn();
            }
            */
            nn = NewCharacterNn();

            var recentTestResults = new List<bool>();
            var keepGoing = false;
            var reached80Percent = false;
            var hundredPercentTimes = 0;
            var keepTraining = true;

            int i = 0;
            const int maxIterations = 10000000;
            DateTime lastDisplayedAt = DateTime.MinValue;
            while (i <= maxIterations) {
                var charCode = RandomCharacter();
                var targetCat = Categorize(charCode);

                CharacterToInputs(charCode, nn.InputLayer);
                CategoryToArray(targetCat, nn.TrainingOutputs);

                nn.FeedForward();
                var predictedCat = (categoryEnum)nn.Classify();
                var success = predictedCat == targetCat;
                recentTestResults.Add(success);
                if (keepTraining) nn.Backpropagate();
                //var l1Rate = nn.CalculateWeightL1() / nn.CountInputWeights();
                //var l2Rate = nn.CalculateWeightL2() / nn.CountInputWeights();

                float percentSuccesses = 0;
                if (recentTestResults.Count > 5000) {
                    recentTestResults.RemoveAt(0);
                    int successes = 0;
                    for (int j = 0; j < recentTestResults.Count; j++) {
                        if (recentTestResults[j]) successes++;
                    }
                    percentSuccesses = 100f * successes / recentTestResults.Count;
                }

                if (percentSuccesses == 100f) {
                    hundredPercentTimes++;
                } else {
                    hundredPercentTimes = 0;
                }

                if (!reached80Percent && percentSuccesses >= 80) {
                    reached80Percent = true;
                } else if (reached80Percent && percentSuccesses < 50) {
                    Console.WriteLine("Training process went from above 80% to below 50% success rate. Restarting from scratch");
                    Console.Beep();
                    //Console.ReadLine();

                    nn = NewCharacterNn();
                    i = 0;
                    reached80Percent = false;
                    keepGoing = false;
                    recentTestResults.Clear();
                } else if (i == 500000 && percentSuccesses < 50) {
                    Console.WriteLine("Training process below 50% success rate after a long time. Restarting from scratch");
                    Console.Beep();
                    //Console.ReadLine();

                    nn = NewCharacterNn();
                    i = 0;
                    reached80Percent = false;
                    keepGoing = false;
                    recentTestResults.Clear();
                } else if (i > 10000 && percentSuccesses < 20) {
                    Console.WriteLine("Training process below 20% success rate. Restarting from scratch");
                    Console.Beep();
                    //Console.ReadLine();

                    nn = NewCharacterNn();
                    i = 0;
                    reached80Percent = false;
                    keepGoing = false;
                    recentTestResults.Clear();
                    //} else if (i == maxIterations - 1 && percentSuccesses < 99) {
                } else if (i == maxIterations - 1 && percentSuccesses < 100) {
                    Console.WriteLine("Training process below 99% success rate after full run. Restarting from scratch");
                    Console.Beep();
                    //Console.ReadLine();

                    nn = NewCharacterNn();
                    i = 0;
                    reached80Percent = false;
                    keepGoing = false;
                    recentTestResults.Clear();
                }

                // Skip most status updates
                if (DateTime.Now.Subtract(lastDisplayedAt).TotalSeconds >= 0.25) {
                    lastDisplayedAt = DateTime.Now;
                    Console.WriteLine(
                        i.ToString("0") + "   " +
                        RenderPercent(percentSuccesses) + " " +
                        percentSuccesses.ToString("0.00") + "%   " +
                        //l2Rate.ToString("0.00") + "  " +
                        //l2Rate.ToString("0.00") + "  "
                    /*charCode + "  " +
                    predictedCat + "  " +
                    (success ? "" : "---- WRONG ----")*/
                        ""
                    );
                }

                if (!keepGoing && hundredPercentTimes >= 1000) {
                    Console.WriteLine("Stable at 100% success for a while. Continue anyway?");
                    Console.Beep();
                    Console.ReadLine();
                    keepGoing = true;
                    keepTraining = false;
                    i = 0;
                } else if (keepGoing && hundredPercentTimes == 0) {
                    keepGoing = false;
                }

                i++;
            }

            //File.WriteAllText(filePath, nn.ToJson());
        }

        static NeuralNetwork NewCharacterNn() {
            var nn = new NeuralNetwork();
            nn.AddLayer(charBitCount);
            nn.AddLayer(10, true, ActivationFunctionEnum.LReLU, 0.01f);
            nn.AddLayer(categoryCount, true, ActivationFunctionEnum.LReLU, 0.01f);
            return nn;
        }

        enum categoryEnum {
            None = -1,
            Whitespace,
            Symbol,
            Letter,
            Digit,
        }

        const int categoryCount = 4;
        const int charBitCount = 7;  // First 7 bits of ASCII code
        const int characterCount = 126 - 32 + 1;  // Space - Tilde (95)

        static char RandomCharacter() {
            var charIndex = NeuralNetwork.NextRandomInt(0, characterCount);
            return (char)(charIndex + 32);
        }

        /// <summary>
        /// Returns the category for the given (ASCII) character code
        /// </summary>
        static categoryEnum Categorize(int charCode) {
            switch (charCode) {
                case 32:
                    return categoryEnum.Whitespace;
                case var n when n >= 48 && n <= 57:
                    return categoryEnum.Digit;
                case var n when n >= 65 && n <= 90:
                    return categoryEnum.Letter;
                case var n when n >= 97 && n <= 122:
                    return categoryEnum.Letter;
                default:
                    return categoryEnum.Symbol;
            }
        }

        static void CategoryToArray(categoryEnum cat, float[] targetArray) {
            for (int i = 0; i < categoryCount; i++) {
                targetArray[i] = 0;
            }
            targetArray[(int)cat] = 1;
        }

        /// <summary>
        /// Transform the character into a 7-bit array of 0 and 1 values
        /// </summary>
        static void CharacterToInputs(char charCode, Layer inputLayer) {
            uint charBits = charCode;
            for (int i = 0; i < charBitCount; i++) {
                uint bit = charBits & 1;
                inputLayer.Neurons[i].Output = bit;
                charBits = charBits >> 1;
            }
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
