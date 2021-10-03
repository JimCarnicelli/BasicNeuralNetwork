using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;

namespace BasicNeuralNetwork {
    public class MnistDigits : IDisposable {
        private const int imgWidth = 28;
        private const int imgHeight = 28;
        private DirectBitmap digitsMontage;
        private float[][] digitInputs;
        private int[] digitOutputs;
        private List<bool> successes = new List<bool>();

        public NeuralNetwork nn;
        public int imgCount;

        public void Dispose() {
            digitsMontage.Dispose();
            digitInputs = null;
            digitOutputs = null;
        }

        public void StartTesting() {
            successes.Clear();
        }

        public void TestIteration(int imgIndex) {
            int expected = digitOutputs[imgIndex];
            nn.SetInputs(digitInputs[imgIndex]);
            nn.FeedForward();
            int predicted = nn.Classify();
            successes.Add(predicted == expected);
        }

        public void StartTraining() {
            nn = new NeuralNetwork(
                new int[] {
                    imgWidth * imgHeight,
                    100,
                    10,  // 0 - 9 digits
                },
                ActivationFunctionEnum.TanH,
                true  // With training
            );
            nn.SetAllLearningRates(0.000001f);
            nn.SetAllLambdas(0, 0);
            successes.Clear();
        }

        public void TrainIteration(int imgIndex) {
            int expected = digitOutputs[imgIndex];
            nn.SetInputs(digitInputs[imgIndex]);
            nn.SetTrainingOutputs(expected);
            nn.FeedForward();
            int predicted = nn.Classify();
            nn.FeedBack();

            successes.Add(predicted == expected);
            if (successes.Count > 10000) successes.RemoveAt(0);
        }

        public float PercentCorrect() {
            if (successes.Count == 0) return 0;
            int successCount = 0;
            for (int i = 0; i < successes.Count; i++) {
                if (successes[i]) successCount++;
            }
            return (float)successCount / successes.Count;
        }

        public void LoadImage(string path) {
            digitsMontage = new DirectBitmap(path);

            int imagesAcross = digitsMontage.Width / imgWidth;
            int imagesDown = digitsMontage.Height / imgHeight;
            imgCount = imagesDown * imagesAcross;
            digitInputs = new float[imgCount][];
            digitOutputs = new int[imgCount];

            for (int imgDown = 0; imgDown < imagesDown; imgDown++) {
                for (int imgAcross = 0; imgAcross < imagesAcross; imgAcross++) {
                    float[] inputs = new float[imgWidth * imgHeight];
                    int label = -1;

                    for (int y = 0; y < imgHeight; y++) {
                        for (int x = 0; x < imgWidth; x++) {
                            Color c = digitsMontage.GetPixel(
                                imgAcross * imgWidth + x,
                                imgDown * imgHeight + y
                            );
                            int value = c.B;
                            if (x == 0 && y == 0) {
                                label = value;
                                value = 255;
                            }
                            value = 255 - value;  // Invert grayscale
                            inputs[y * imgWidth + x] = value / 255f;
                        }
                    }

                    digitInputs[imgDown * imagesAcross + imgAcross] = inputs;
                    digitOutputs[imgDown * imagesAcross + imgAcross] = label;
                }
            }
        }

        public void ConvertImage(string pixelFile, string labelFile, string outputFile, int numImages, int imagesAcross) {
            var sourceImage = LoadOriginalData(pixelFile, labelFile, numImages);

            int imagesDown = numImages / imagesAcross;

            var newImage = new DirectBitmap(imagesAcross * imgWidth, imagesDown * imgHeight);
            for (int imgDown = 0; imgDown < imagesDown; imgDown++) {
                for (int imgAcross = 0; imgAcross < imagesAcross; imgAcross++) {
                    var srcImg = sourceImage[imgDown * imagesAcross + imgAcross];
                    for (int y = 0; y < imgHeight; y++) {
                        for (int x = 0; x < imgWidth; x++) {
                            var cb = 255 - (int)srcImg.pixels[y][x];
                            var c = Color.FromArgb(255, cb, cb, cb);

                            if (x == 0 && y == 0) {
                                // Paint the label (0 - 9) into the top left corner pixel as a mostly red dot
                                cb = srcImg.label;
                                c = Color.FromArgb(255, 255, 0, cb);
                            }

                            newImage.SetPixel(
                                imgAcross * imgWidth + x,
                                imgDown * imgHeight + y,
                                c
                            );
                        }
                    }
                }
            }
            newImage.Bitmap.Save(outputFile, ImageFormat.Png);
            newImage.Dispose();
        }

        public class DigitImage {
            public int width; // 28
            public int height; // 28
            public byte[][] pixels; // 0(white) - 255(black)
            public byte label; // '0' - '9'
            public DigitImage(int width, int height,
              byte[][] pixels, byte label) {
                this.width = width; this.height = height;
                this.pixels = new byte[height][];
                for (int i = 0; i < this.pixels.Length; ++i)
                    this.pixels[i] = new byte[width];
                for (int i = 0; i < height; ++i)
                    for (int j = 0; j < width; ++j)
                        this.pixels[i][j] = pixels[i][j];
                this.label = label;
            }
        }

        // Adapted from: https://docs.microsoft.com/en-us/archive/msdn-magazine/2014/june/test-run-working-with-the-mnist-image-recognition-data-set
        public DigitImage[] LoadOriginalData(string pixelFile, string labelFile, int numImages) {
            DigitImage[] result = new DigitImage[numImages];
            byte[][] pixels = new byte[28][];
            for (int i = 0; i < pixels.Length; ++i)
                pixels[i] = new byte[28];
            FileStream ifsPixels = new FileStream(pixelFile, FileMode.Open);
            FileStream ifsLabels = new FileStream(labelFile, FileMode.Open);
            BinaryReader brImages = new BinaryReader(ifsPixels);
            BinaryReader brLabels = new BinaryReader(ifsLabels);
            int magic1 = brImages.ReadInt32(); // stored as big endian
            magic1 = ReverseBytes(magic1); // convert to Intel format
            int imageCount = brImages.ReadInt32();
            imageCount = ReverseBytes(imageCount);
            int numRows = brImages.ReadInt32();
            numRows = ReverseBytes(numRows);
            int numCols = brImages.ReadInt32();
            numCols = ReverseBytes(numCols);
            int magic2 = brLabels.ReadInt32();
            magic2 = ReverseBytes(magic2);
            int numLabels = brLabels.ReadInt32();
            numLabels = ReverseBytes(numLabels);
            for (int di = 0; di < numImages; ++di) {
                for (int i = 0; i < 28; ++i) // get 28x28 pixel values
                {
                    for (int j = 0; j < 28; ++j) {
                        byte b = brImages.ReadByte();
                        pixels[i][j] = b;
                    }
                }
                byte lbl = brLabels.ReadByte(); // get the label
                DigitImage dImage = new DigitImage(28, 28, pixels, lbl);
                result[di] = dImage;
            } // Each image
            ifsPixels.Close(); brImages.Close();
            ifsLabels.Close(); brLabels.Close();
            return result;
        }

        public static int ReverseBytes(int v) {
            byte[] intAsBytes = BitConverter.GetBytes(v);
            Array.Reverse(intAsBytes);
            return BitConverter.ToInt32(intAsBytes, 0);
        }

    }
}
