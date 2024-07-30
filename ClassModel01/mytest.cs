using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;
using ClassModel01;
using static TorchSharp.torch.nn;
using System.Diagnostics;
using OpenCvSharp;
using System.Runtime.InteropServices;
using ResNet = ClassModel01.ResNet;

namespace Mytest
{
    internal class mytest
    {
        /// <summary>
        /// 查看Dataloader是否正常加载
        /// </summary>
        public void Dataloadtest()
        {
            List<string> list_images = new List<string>();
            List<int> list_labels = new List<int>();
            string path = @"D:\project\YoloTorchSharp\flower_photos\train";
            string type = ".jpg"; // or ".png", ".jpeg", etc.
            DataSetClc dataset = new DataSetClc(path, type);

            Device device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            var train_loder = new DataLoader(dataset, 16, device: device, num_worker: 1, shuffle: true);

            foreach (var batch in train_loder)
            {
                var data = batch["data"];
                var labels = batch["label"];

                // 在此处理数据和标签，例如训练模型
                Console.WriteLine("Data batch shape: " + string.Join(", ", data.shape));
                Console.WriteLine("Label batch shape: " + string.Join(", ", labels.shape));

                var dataArray = data.data<float>().ToArray();
                Console.WriteLine("Data batch values: " + string.Join(", ", dataArray.Take(10)));

                // 打印标签张量的值
                var labelArray = labels.data<Int32>().ToArray();
                Console.WriteLine("Label batch values: " + string.Join(", ", labelArray));
            }
        }

        /// <summary>
        /// 打印模型参数
        /// </summary>
        /// <param name="module"></param>
        public static void PrintModelParameters(Module module)
        {
            foreach (var param in module.parameters())
            {
                Console.WriteLine($"{param}: {param.shape}");
            }
        }
        /// <summary>
        /// 打印模型结构
        /// </summary>
        /// <param name="module"></param>
        /// <param name="indent"></param>
        public static void PrintModelStructure(Module module, string indent = "")
        {
            Console.WriteLine($"{indent}{module.GetType().Name}");
            foreach (var submodule in module.modules())
            {
                PrintModelStructure(submodule, indent + "  ");
            }
        }
    }
}
