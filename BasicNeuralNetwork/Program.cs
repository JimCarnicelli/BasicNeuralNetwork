using System;
using System.Collections.Generic;
using System.IO;

namespace BasicNeuralNetwork {
    class Program {

        static void Main(string[] args) {

            //RunXorDemo();
            //RunAsciiDemo();

            //ConvertDigitsImages();
            RunDigitsTraining();
            //RunDigitsTest();

            Console.WriteLine("Done");
            Console.Beep();
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
        }


        #region ASCII character demo


        static void RunAsciiDemo() {
            NeuralNetwork nn;

            const bool loadFromFile = true;
            var filePath = DataDirectory() + "Character Neural Network.json";

            if (loadFromFile && File.Exists(filePath)) {
                var json = File.ReadAllText(filePath);
                nn = new NeuralNetwork();
                nn.FromJson(json);
            } else {
                nn = NewCharacterNn();
            }

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
                    File.WriteAllText(filePath, nn.ToJson());
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


        #endregion


        #region Digit image recognition demo


        static string digitsDemoFilePath = DataDirectory() + "MNIST digits neural network.json";

        static void RunDigitsTraining() {
            var mnist = new MnistDigits();
            mnist.LoadImage(DataDirectory() + "Mnist images/Training images.png");

            bool loadFromFile = false;

            mnist.StartTraining(loadFromFile, digitsDemoFilePath);

            const int maxIterations = 1000000;
            int i = 0;
            float bestPercentCorrect = 0;
            float bestOfBestPercentCorrect = 0;
            float hundredPercentInARow = 0;
            DateTime lastDisplayedAt = DateTime.MinValue;

            while (i < maxIterations) {
                int imgIndex = NeuralNetwork.NextRandomInt(0, mnist.imgCount);
                mnist.TrainIteration(imgIndex);

                float percentCorrect = mnist.PercentCorrect();

                if (DateTime.Now.Subtract(lastDisplayedAt).TotalSeconds >= 0.5) {
                    lastDisplayedAt = DateTime.Now;
                    Console.WriteLine(
                        RightJustify(i.ToString("#,##0"), 9) +
                        RightJustify((percentCorrect * 100).ToString("0.0") + "% ", 12) +
                        RenderPercent(percentCorrect * 100)
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
                        File.WriteAllText(digitsDemoFilePath, json);

                        RunDigitsTest();

                        Console.Beep();
                        Console.ReadLine();
                    }
                }

                if (i > 1000 && percentCorrect < bestPercentCorrect * 0.7) {
                    Console.WriteLine("Dropped by 30% from best (" + (100 * percentCorrect).ToString("0.0") + "%). Restarting.");
                    Console.WriteLine("Best of best: " + (100 * bestOfBestPercentCorrect).ToString("0.0") + "%");
                    //Console.Beep();
                    mnist.StartTraining(false, null);
                    bestPercentCorrect = 0;
                    i = 0;
                }

                if (i == maxIterations - 1 && percentCorrect < 0.90f) {
                    Console.WriteLine("Still below 90%. Restarting.");
                    Console.WriteLine("Best of best: " + (100 * bestOfBestPercentCorrect).ToString("0.0") + "%");
                    //Console.Beep();
                    mnist.StartTraining(false, null);
                    bestPercentCorrect = 0;
                    i = 0;
                }

                i++;
            }

            {
                string json = mnist.nn.ToJson();
                File.WriteAllText(digitsDemoFilePath, json);
                RunDigitsTest();
            }

            mnist.Dispose();
        }

        static void ConvertDigitsImages() {
            var mnist = new MnistDigits();
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
        }

        static void RunDigitsTest() {
            Console.WriteLine("Testing against test image set...");

            var mnistTest = new MnistDigits();
            mnistTest.LoadImage(DataDirectory() + "Mnist images/Test images.png");
            mnistTest.nn = new NeuralNetwork();

            string json = File.ReadAllText(digitsDemoFilePath);
            mnistTest.nn.FromJson(json);
            mnistTest.StartTesting();
            for (int j = 0; j < mnistTest.imgCount; j++) {
                mnistTest.TestIteration(j);
            }
            var percentCorrect = mnistTest.PercentCorrect();
            Console.WriteLine("Tested against test set: " + (100 * percentCorrect).ToString("0.0") + "% correct.");
        }


        #endregion


        static string DataDirectory() {
            string path = Environment.CurrentDirectory + "/Data/";
            path = @"G:\My Drive\Ventures\MsDev\BasicNeuralNetwork\Data\";
            if (!Directory.Exists(path)) {
                Directory.CreateDirectory(path);
            }
            return path;
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
