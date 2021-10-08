using System;
using System.Collections.Generic;
using System.IO;

namespace BasicNeuralNetworkV1 {

    class ProgramV1 {

        static void OldMain(string[] args) {

            //CodingBackProp.BackPropProgram.xMain(args);

            //TrainDigits();
            TrainCharacters();
            //TrainXor();

            Console.WriteLine("Done");
            Console.Beep();
            Console.ReadLine();
        }

        static void TrainDigits() {
            var mnist = new MnistDigits();
            mnist.LoadImage(DataDirectory() + "Mnist images/Training images.png");

            var mnistTest = new MnistDigits();
            mnistTest.LoadImage(DataDirectory() + "Mnist images/Test images.png");

            mnist.StartTraining();
            int inputWeights = mnist.nn.CountInputWeights();

            const int maxIterations = 1000000;
            int i = 0;
            float bestPercentCorrect = 0;
            float bestOfBestPercentCorrect = 0;
            float hundredPercentInARow = 0;
            DateTime lastDisplayedAt = DateTime.MinValue;

            while (i < maxIterations) {
                int imgIndex = mnist.nn.random.Next(0, mnist.imgCount);
                mnist.TrainIteration(imgIndex);

                float percentCorrect = mnist.PercentCorrect();
                float avgL1 = mnist.nn.CalculateWeightL1() / inputWeights;
                float avgL2 = mnist.nn.CalculateWeightL2() / inputWeights;

                if (DateTime.Now.Subtract(lastDisplayedAt).TotalSeconds >= 0.25) {
                    lastDisplayedAt = DateTime.Now;
                    Console.WriteLine(
                        i + "   " +
                        RenderPercent(percentCorrect * 100f) + " " +
                        (percentCorrect * 100f).ToString("0.00") + "%    " +
                        avgL1.ToString("#,##0.00") + "  " +
                        avgL2.ToString("#,##0.00") + "  "
                    );
                }

                if (i > 100) {
                    if (percentCorrect > bestPercentCorrect) {
                        bestPercentCorrect = percentCorrect;
                        if (bestPercentCorrect > bestOfBestPercentCorrect) {
                            bestOfBestPercentCorrect = bestPercentCorrect;
                        }
                    }

                    if (percentCorrect == 100) {
                        hundredPercentInARow++;
                    } else {
                        hundredPercentInARow = 0;
                    }
                    if (hundredPercentInARow > 1000) {
                        Console.WriteLine("100% correct for past 1k iterations at " + DateTime.Now.ToString());
                        Console.Beep();

                        string json = mnist.nn.ToJson();
                        var filePath = DataDirectory() + "MNIST digits neural network.json";
                        File.WriteAllText(filePath, json);

                        mnistTest.nn = new NeuralNetworkV1(json);
                        mnistTest.StartTesting();
                        for (int j = 0; j < mnistTest.imgCount; j++) {
                            mnistTest.TestIteration(j);
                        }
                        percentCorrect = mnistTest.PercentCorrect();
                        Console.WriteLine("Tested against test set: " + (100 * percentCorrect).ToString("0.0") + "% correct.");

                        Console.Beep();
                        Console.ReadLine();
                    }
                }

                if (i > 1000 && percentCorrect < bestPercentCorrect * 0.7) {
                    Console.WriteLine("Dropped by 30% from best (" + (100 * percentCorrect).ToString("0.0") + "%). Restarting.");
                    Console.WriteLine("Best of best: " + (100 * bestOfBestPercentCorrect).ToString("0.0") + "%");
                    //Console.Beep();
                    mnist.StartTraining();
                    bestPercentCorrect = 0;
                    i = 0;
                }

                if (i == maxIterations - 1 && percentCorrect < 0.90f) {
                    Console.WriteLine("Still below 90%. Restarting.");
                    Console.WriteLine("Best of best: " + (100 * bestOfBestPercentCorrect).ToString("0.0") + "%");
                    //Console.Beep();
                    mnist.StartTraining();
                    bestPercentCorrect = 0;
                    i = 0;
                }

                i++;
            }

            /*
            mnist.ConvertImage(
                DataDirectory() + "Mnist images/train-images.idx3-ubyte",
                DataDirectory() + "Mnist images/train-labels.idx1-ubyte",
                DataDirectory() + "Mnist images/Training images.png",
                60000,
                300
            );
            mnist.ConvertImage(
                DataDirectory() + "Mnist images/t10k-images.idx3-ubyte",
                DataDirectory() + "Mnist images/t10k-labels.idx1-ubyte",
                DataDirectory() + "Mnist images/Test images.png",
                10000,
                100
            );
            */

            mnist.Dispose();
        }

        static void TrainCharacters() {

            NeuralNetworkV1 nn;

            const bool loadFromFile = false;
            var filePath = DataDirectory() + "Character Neural Network.json";

            if (loadFromFile && File.Exists(filePath)) {
                var json = File.ReadAllText(filePath);
                nn = new NeuralNetworkV1(json, true);

            } else {
                nn = NewCharacterNn();
            }

            var recentTestResults = new List<bool>();
            var keepGoing = false;
            var reached80Percent = false;
            var hundredPercentTimes = 0;
            var keepTraining = true;

            int i = 0;
            const int maxIterations = 500000;
            DateTime lastDisplayedAt = DateTime.MinValue;
            while (i <= maxIterations) {
                var charCode = RandomCharacter(nn);
                var targetCat = Categorize(charCode);

                CharacterToArray(charCode, nn.inputs);
                CategoryToArray(targetCat, nn.trainingOutputs);

                nn.FeedForward();
                var predictedCat = OutputToCategory(nn.outputs);
                var success = predictedCat == targetCat;
                recentTestResults.Add(success);
                if (keepTraining) nn.FeedBack();
                var l1Rate = nn.CalculateWeightL1() / nn.CountInputWeights();
                var l2Rate = nn.CalculateWeightL2() / nn.CountInputWeights();

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
                        l2Rate.ToString("0.00") + "  " +
                        l2Rate.ToString("0.00") + "  "
                        /*charCode + "  " +
                        predictedCat + "  " +
                        (success ? "" : "---- WRONG ----")*/
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

            File.WriteAllText(filePath, nn.ToJson());
        }

        static NeuralNetworkV1 NewCharacterNn() {
            var nn = new NeuralNetworkV1(
                new int[] {
                    charBitCount,
                    8,
                    categoryCount,
                },
                ActivationFunctionEnum.Sigmoid,
                true
            );
            nn.SetAllLearningRates(0.005f);
            nn.SetAllLambdas(0f, 0f);
            return nn;
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

        static char RandomCharacter(NeuralNetworkV1 nn) {
            var charIndex = nn.random.Next(0, characterCount);
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
        static void CharacterToArray(char charCode, float[] targetArray) {
            uint charBits = charCode;
            for (int i = 0; i < charBitCount; i++) {
                uint bit = charBits & 1;
                targetArray[i] = bit;
                charBits = charBits >> 1;
            }
        }

        static categoryEnum OutputToCategory(float[] outputs) {
            float maxValue = 0;
            int maxIndex = -1;
            for (int i = 0; i < categoryCount; i++) {
                float value = outputs[i];
                if (value > maxValue) {
                    maxIndex = i;
                    maxValue = value;
                }
            }
            if (maxValue < 0.5) return categoryEnum.None;
            return (categoryEnum)maxIndex;
        }



        static void TrainXor() {

            NeuralNetworkV1 nn;

            const bool loadFromFile = false;
            var filePath = DataDirectory() + "Xor Neural Network.json";

            if (loadFromFile && File.Exists(filePath)) {
                var json = File.ReadAllText(filePath);
                nn = new NeuralNetworkV1(json, true);

            } else {
                nn = new NeuralNetworkV1(
                    new int[] { 2, 2, 1 },
                    ActivationFunctionEnum.TanH,
                    true
                );
            }

            // All input cases
            float[][] inputs = {
                new float[]{ 0, 0 },
                new float[]{ 0, 1 },
                new float[]{ 1, 0 },
                new float[]{ 1, 1 }
            };

            // Target output for each input case
            float[] results = { 0, 1, 1, 0 };

            const int maxIterations = 10000;
            int lastErrorAt = 0;
            for (int i = 0; i < maxIterations; i++) {

                int j = nn.random.Next(0, 4);
                inputs[j].CopyTo(nn.inputs, 0);
                nn.trainingOutputs[0] = results[j];

                nn.FeedForward();
                nn.FeedBack();
                int o = nn.outputs[0] >= 0.5 ? 1 : 0;
                bool success = o == results[j];
                if (!success) lastErrorAt = i;
                Console.WriteLine(
                    i + "   " +
                    nn.inputs[0].ToString("0") + "  xor  " +
                    nn.inputs[1].ToString("0") + "  =  " +
                    nn.outputs[0].ToString("0.000") + "   " +
                    (success ? "        " : "--------") + "   " +
                    nn.CalculateWeightL2().ToString("0.00")
                );
            }

            File.WriteAllText(filePath, nn.ToJson());

            Console.WriteLine("Last error at " + lastErrorAt);
        }

        static string DataDirectory() {
            string path = Environment.CurrentDirectory + "/Data/";
            path = @"G:\My Drive\Ventures\MsDev\BasicNeuralNetwork\Data\";
            if (!Directory.Exists(path)) {
                Directory.CreateDirectory(path);
            }
            return path;
        }

    }
}
