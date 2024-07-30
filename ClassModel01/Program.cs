using OpenCvSharp;
using OpenCvSharp.Dnn;
using OpenCvSharp.ML;
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
using TorchSharp;
using TorchSharp.Data;
using TorchSharp.Modules;
using static System.Net.Mime.MediaTypeNames;
using static Tensorboard.LogMessage.Types;
using static Tensorboard.Summary.Types;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.utils;

namespace ClassModel01
{
    internal class Program
    {
        static void Main(string[] args)
        {
            flowertrain.flowerTrain();
        }

    }
}
