using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicNeuralNetwork {
    class Program {

        static void Main(string[] args) {

            var nn = new NeuralNetwork();
            nn.AddLayer(2);
            nn.AddLayer(2);
            nn.AddLayer(1);

            Console.WriteLine("Done");
            Console.Beep();
            Console.ReadLine();
        }

    }
}
