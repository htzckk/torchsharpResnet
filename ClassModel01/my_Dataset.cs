using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.utils.data;
using static TorchSharp.torch;
using TorchSharp;
using System.Runtime.InteropServices;
using OpenCvSharp.Aruco;

namespace ClassModel01
{

    /// <summary>
    /// Dataset类
    /// </summary>
    internal class DataSetClc : torch.utils.data.Dataset
    {
        private List<string> imagePaths = new List<string>(); // 存储图像路径的列表
        private List<int> labels = new List<int>(); // 存储标签的列表
        private int numClasses; //类别数
        private Dictionary<string,int> class2idx = new Dictionary<string,int>();

        /// <summary>
        /// 构造函数，传入图像目录和图像类型
        /// </summary>
        /// <param name="imageDir"></param>
        /// <param name="type"></param>
        public DataSetClc(string imageDir, string type)
        {
            //LoadDataFromFolder(imageDir, type, imagePaths, labels, ref numClasses);
            LoadDataFromFolder1(imageDir, type, imagePaths, labels, ref numClasses,ref class2idx);

        }

        /// <summary>
        /// 从文件夹加载数据的方法
        /// </summary>
        /// <param name="path"></param>
        /// <param name="type"></param>
        /// <param name="list_images"></param>
        /// <param name="list_labels"></param>
        /// <param name="label"></param>
        public void LoadDataFromFolder(string path, string type, List<string> list_images, List<int> list_labels, ref int label)
        {
            if (!Directory.Exists(path))
            {
                return;
            }

            var files = Directory.GetFiles(path, "*.*", SearchOption.AllDirectories);
            Console.WriteLine("lenth" + files.Length.ToString());
            var directories = Directory.GetDirectories(path);

            foreach (var file in files)
            {
                if (file.EndsWith(type, StringComparison.OrdinalIgnoreCase))
                {
                    list_images.Add(file);
                    list_labels.Add(label);
                }
            }

            foreach (var dir in directories)
            {
                if (dir == "." || dir == "..")
                {
                    continue;
                }
                label++;
                LoadDataFromFolder(dir, type, list_images, list_labels, ref label);
            }
        }

        /// <summary>
        /// 从文件夹加载数据的方法改进版
        /// </summary>
        /// <param name="path"></param>
        /// <param name="type"></param>
        /// <param name="list_images"></param>
        /// <param name="list_labels"></param>
        /// <param name="label"></param>
        /// <param name="class2idx"></param>
        public void LoadDataFromFolder1(string path, string type, List<string> list_images, List<int> list_labels, ref int label,ref Dictionary<string,int> class2idx)
        {
            if (!Directory.Exists(path))
            {
                Console.WriteLine("输入数据集路径不正确");
                return;
            }

            int label_idx = 0;
            var directories = Directory.GetDirectories(path);

            foreach (var dir in directories)
            {
                if (dir == "." || dir == "..")
                {
                    continue;
                }
                var files = Directory.GetFiles(dir, "*.*", SearchOption.AllDirectories);
                string directoryName = Path.GetFileName(dir.TrimEnd(Path.DirectorySeparatorChar));

                if (files.Length == 0)
                {
                    Console.WriteLine($"{dir}" + "中没有图像");
                    return;
                }
                foreach (var file in files)
                {
                    if (file.EndsWith(type, StringComparison.OrdinalIgnoreCase))
                    {
                        list_images.Add(file);
                        list_labels.Add(label_idx);
                    }
                }
                class2idx[directoryName] = label_idx;
                label++;
                label_idx++;
            }
        }

        /// <summary>
        /// 获取指定索引的张量
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public override Dictionary<string, Tensor> GetTensor(long index)
        {
            var tensorDataDic=new Dictionary<string, Tensor>();
            var imagePath = imagePaths[(int)index];
            Mat image = Cv2.ImRead(imagePath); // 使用 OpenCvSharp 读取图像
            Cv2.Resize(image, image, new OpenCvSharp.Size(224, 224)); // 调整图像大小为 448x448
            // Create TorchSharp tensors
            //float[] imageArray = MatToFloatArray(image);
            //var imgTensor = torch.tensor(imageArray, new long[] { 3, 224, 224 });

            // 转换图像到浮点类型
            image.ConvertTo(image, MatType.CV_32FC3);
            // 将图像归一化到 [0, 1]
            image /= 255.0;
            // 获取图像数据
            var imageData = new float[image.Rows * image.Cols * image.Channels()];
            Marshal.Copy(image.Data, imageData, 0, imageData.Length);

            // 创建 TorchSharp 张量
            var imgTensor = torch.tensor(imageData, new long[] { 224, 224, 3 }, dtype: torch.float32).permute(2, 0, 1); // 转换为 C x H x W 格式

            // 创建标签张量
            var labelTensor = torch.tensor(labels[(int)index], dtype: torch.int64);

            tensorDataDic.Add("data", imgTensor);
            tensorDataDic.Add("label", labelTensor);

            return tensorDataDic; // 返回图像张量和标签张量
        }

        /// <summary>
        /// 获取数据集的大小
        /// </summary>
        public override long Count => imagePaths.Count;

        /// <summary>
        /// Mat数据类型转成Array,归一化
        /// </summary>
        /// <param name="mat"></param>
        /// <returns></returns>
        private float[] MatToFloatArray(Mat mat)
        {
            int width = mat.Width;
            int height = mat.Height;
            int channels = mat.Channels();
            float[] result = new float[width * height * channels];
            // Convert Mat to float array
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Vec3b color = mat.At<Vec3b>(y, x);
                    result[(y * width + x) * 3 + 0] = color.Item0 / 255f; // B
                    result[(y * width + x) * 3 + 1] = color.Item1 / 255f; // G
                    result[(y * width + x) * 3 + 2] = color.Item2 / 255f; // R
                }
            }
            return result;
        }
    }
    
}