#ifndef __FACETHINK_API_TENSORRT_HPP__
#define __FACETHINK_API_TENSORRT_HPP__

#include <cuda_runtime_api.h>

#include <cassert>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"
#include "NvUtils.h"
#include "common.h"
#include "yololayer.h"

#define MAX_WORKSPACE (1 << 30)

#define RETURN_AND_LOG(ret, severity, message)                               \
  do {                                                                       \
    std::string error_message = "sample_uff_mnist: " + std::string(message); \
    gLogger.log(ILogger::Severity::k##severity, error_message.c_str());      \
    return (ret);                                                            \
  } while (0)


namespace facethink {
    namespace tensorrt {
        using namespace nvuffparser;
        using namespace nvinfer1;

        std::string toStr(Dims dim);

        class Tensorrt {
        public:
            Tensorrt();
            ~Tensorrt();

            // 加载模型
            void loadModel(const std::string& model_file, const std::vector<std::string>& input_names, const std::vector<std::string>& output_names);

            // 模型推理
            void doInference(float* input, int batch_size, std::vector<int> input_dims, std::vector<float*>& outputs, std::vector<Dims>& output_dims);

        private:
            // 加载UFF模型
            void loadUFFModel(const std::string& model_file);

            // 加载ONNX模型
            void loadONNXModel(const std::string model_file);

            // 加载TRT模型
            void loadTRTModel(const std::string& model_file);

            // 写入模型文件
            void writeGIEModel(IHostMemory* gie_odel_tream, const std::string& model_file);

        private:
            // 模型输入的尺寸
            //std::vector<Dims> input_dims_;
            //std::vector<int64_t> input_sizes_;
            std::vector<std::string> input_names_;
            //std::vector<int> input_indexes_;

            // 模型输出的尺寸
            //std::vector<Dims> output_dims_;
            //std::vector<int64_t> output_sizes_;
            std::vector<std::string> output_names_;
            //std::vector<float*> output_ptrs_;
            //std::vector<int> output_indexes_;

            // 模型输入的信息
            int max_batch_ = 1;
            int input_channel_ = 12;
            int input_height_ = 192;
            int input_width_ = 320;
            IExecutionContext* context_;
            ICudaEngine* engine_ = nullptr;
            //std::vector<void*> gpu_buffers_;
            //cudaStream_t  stream_;
        };
    }  // namespace tensorrt
}  // namespace facethink

#endif
