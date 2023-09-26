#ifndef __FACETHINK_API_CLS_TIMU_HPP__
#define __FACETHINK_API_CLS_TIMU_HPP__

#include "opencv2/opencv.hpp"

#ifdef WIN32
#ifdef DLL_EXPORTS
#define EXPORT_CLASS __declspec(dllexport)
#define EXPORT_API extern "C" __declspec(dllexport)
#define EXPORT_CLASS_API

#else
#define EXPORT_CLASS __declspec(dllimport)
#define EXPORT_API extern "C" __declspec(dllimport)
#endif
#else
#define EXPORT_CLASS
#define EXPORT_API extern "C" __attribute__((visibility("default")))
#define EXPORT_CLASS_API __attribute__((visibility("default")))
#endif

namespace facethink {
class EXPORT_CLASS ClsTopic {
 public:

  /**
   * @brief SDK初始化函数，必须先于其他SDK函数之前调用
   * @param mode_path 模型文件路径
   * @param config_path SDK对应的ini配置文件路径，详见config.ini
   * @remarks 初始化函数需要读取模型等文件 
   */
  EXPORT_CLASS_API static ClsTopic* create(const std::string& model_path, const std::string& config_path);


  /**
   * @brief 检测图片的旋转方向
   * @param images 待检测的图片列表，支持如下两种格式：
   *  1.BGR图：img为一维数组，每个元素（字节）表示一个像素点的单通道取值，三个连续元素表示一个像素点的三通道取值，顺序为BGR。
   *  2.RGB图：此时is_rgb_format应设置为true。
   * @param probs 返回每张图片不同方向的概率
   */
  EXPORT_CLASS_API virtual int detection(const std::vector<cv::Mat>& images, std::vector<std::vector<float>>& probs, bool is_rgb_format = false) = 0;

  EXPORT_CLASS_API virtual ~ClsTopic(void);

 protected:
  EXPORT_CLASS_API explicit ClsTopic(void);

 private:
  ClsTopic(const ClsTopic&);
  ClsTopic& operator=(const ClsTopic&);
};

}  // namespace facethink

#endif  //__FACETHINK_API_CLS_TIMU_HPP__