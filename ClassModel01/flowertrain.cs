using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp;
using TorchSharp.Modules;
using System.Diagnostics;

namespace ClassModel01
{
    internal class flowertrain
    {
        /// <summary>
        /// flowerTrain
        /// </summary>
        public static void flowerTrain()
        {
            Device device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            //Device device =  torch.CPU;

            var writer = String.IsNullOrEmpty("./writer") ? null : torch.utils.tensorboard.SummaryWriter("./writer", createRunName: true);

            int _epochs = 10;
            ClassModel01.ResNet model = ClassModel01.ResNet.ResNet34(5, device);
            string trainpath = @"D:\project\YoloTorchSharp\flower_photos\train";
            string valpath = @"D:\project\YoloTorchSharp\flower_photos\val";
            string type = ".jpg"; // or ".png", ".jpeg", etc.

            DataSetClc traindataset = new DataSetClc(trainpath, type);
            DataSetClc valdataset = new DataSetClc(valpath, type);

            var train_loder = new DataLoader(traindataset, 16, device: device, num_worker: 1, shuffle: true);
            var val_loder = new DataLoader(valdataset, 16, device: device, num_worker: 1, shuffle: true);

            model.to(device);
            var optimizer = optim.Adam(model.parameters());
            var scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.7);
            var loss_function = nn.CrossEntropyLoss();

            Stopwatch totalTime = new Stopwatch();
            totalTime.Start();

            for (var epoch = 1; epoch <= _epochs; epoch++)
            {
                TrainOneEpoch(model,optimizer,loss_function,device,train_loder,epoch,16,(int)traindataset.Count);
                Val(model, loss_function, writer, device, val_loder, epoch, (int)valdataset.Count);
            }

            totalTime.Stop();
            Console.WriteLine($"Elapsed time: {totalTime.Elapsed.TotalSeconds:F1} s.");
            Console.WriteLine("Saving model to '{0}'", "model.bin");
            model.save("model.bin");
        }

        /// <summary>
        /// Train
        /// </summary>
        /// <param name="model"></param>
        /// <param name="optimizer"></param>
        /// <param name="loss"></param>
        /// <param name="device"></param>
        /// <param name="dataLoader"></param>
        /// <param name="epoch"></param>
        /// <param name="batchSize"></param>
        /// <param name="size"></param>
        private static void TrainOneEpoch(
            Module<Tensor, Tensor> model,
            optim.Optimizer optimizer,
            Loss<Tensor, Tensor, Tensor> loss,
            Device device,
            DataLoader dataLoader,
            int epoch,
            long batchSize,
            int size)
        {
            model.train();

            int batchId = 1;

            Console.WriteLine($"Epoch: {epoch} start");

            foreach (var trainbatch in dataLoader)
            {
                using (var d = torch.NewDisposeScope())
                {
                    optimizer.zero_grad();

                    var data = trainbatch["data"];
                    var label = trainbatch["label"];
                    var prediction = model.forward(data);
                    var output = loss.forward(prediction, label);

                    output.backward();

                    optimizer.step();

                    if (batchId % 32 == 0)
                    {
                        Console.WriteLine($"\rTrain: epoch {epoch} [{batchId * batchSize} / {size}] Loss: {output.ToSingle():F4}");
                    }
                    batchId++;
                }

            }
        }
        /// <summary>
        /// Val
        /// </summary>
        /// <param name="model"></param>
        /// <param name="loss"></param>
        /// <param name="writer"></param>
        /// <param name="device"></param>
        /// <param name="dataLoader"></param>
        /// <param name="epoch"></param>
        /// <param name="size"></param>

        private static void Val(
           Module<Tensor, Tensor> model,
           Loss<Tensor, Tensor, Tensor> loss,
           TorchSharp.Modules.SummaryWriter writer,
           Device device,
           DataLoader dataLoader,
           int epoch,
           int size)
        {
            model.eval();
            double testLoss = 0;
            int correct = 0;
            foreach (var valbatch in dataLoader)
            {
                using (var d = torch.NewDisposeScope())
                {
                    var data = valbatch["data"];
                    var label = valbatch["label"];
                    var prediction = model.forward(data);
                    var output = loss.forward(prediction, label);
                    testLoss += output.ToSingle();

                    correct += prediction.argmax(1).eq(label).sum().ToInt32();
                }
            }
            Console.WriteLine($"Size: {size}, Total: {size}");

            Console.WriteLine($"\rTest set: Average loss {(testLoss / size):F4} | Accuracy {((double)correct / size):P2}");

            if (writer != null)
            {
                writer.add_scalar("flower/loss", (float)(testLoss / size), epoch);
                writer.add_scalar("flower/accuracy", (float)correct / size, epoch);
            }
        }
    }
}
