using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;

namespace ClassModel01
{
    internal class flowerpredict
    {
        /// <summary>
        /// flowerPredict
        /// </summary>
        public static void flowerPredict()
        {
            string modelpath = "model.bin";
            string imagepath = @"D:\project\YoloTorchSharp\flower_photos\train\daisy\5547758_eea9edfd54_n.jpg";
            Device device = torch.CPU;

            var net = ResNet.ResNet34(5, device);
            net.load(modelpath);
            net.eval();
            var img = Cv2.ImRead(imagepath);
            Cv2.Resize(img, img, new OpenCvSharp.Size(224, 224)); // 调整图像大小为 224x224
            // 转换图像到浮点类型
            img.ConvertTo(img, MatType.CV_32FC3);
            // 将图像归一化到 [0, 1]
            img /= 255.0;
            // 获取图像数据
            var imageData = new float[img.Rows * img.Cols * img.Channels()];
            Marshal.Copy(img.Data, imageData, 0, imageData.Length);

            // 创建 TorchSharp 张量
            var imgTensor = torch.tensor(imageData, new long[] { 224, 224, 3 }, dtype: torch.float32).permute(2, 0, 1); // 转换为 C x H x W 格式
            imgTensor = imgTensor.unsqueeze(0);
            var output = net.forward(imgTensor);
            output = torch.softmax(output, 1);

            var dataArray = output.data<float>().ToArray();
            Console.WriteLine(string.Join(", ", dataArray));
        }
    }
    
}
