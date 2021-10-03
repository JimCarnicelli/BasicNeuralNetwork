using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicNeuralNetwork {
    class Program {

        static void Main(string[] args) {

            var nn = new NeuralNetwork();
            nn.AddLayer(2, true);
            nn.AddLayer(2, true);
            nn.AddLayer(1, true);

            Console.WriteLine("Done");
            Console.Beep();
            Console.ReadLine();
        }

    }
}
