#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <vector>
#include <string>
#include "NvInfer.h"

namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.1f;
    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT * 2];
    };
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 10080;
    static constexpr int CLASS_NUM = 1;
    static constexpr int INPUT_H = 320;
    static constexpr int INPUT_W = 512;

    static constexpr int LOCATIONS = 4;
    struct alignas(float) Detection {
        //center_x center_y w h
        float bbox[LOCATIONS];
        float conf;  // bbox_conf * cls_conf
        float class_id;
    };
}

namespace nvinfer1
{
    class YoloLayerPlugin : public IPluginV2DynamicExt
    {
    public:
        YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, const std::vector<Yolo::YoloKernel>& vYoloKernel);
        YoloLayerPlugin(const void* data, size_t length);
        ~YoloLayerPlugin();

        int getNbOutputs() const override
        {
            return 1;
        }

        //override for dynamic shape plugin
        size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                                const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;

        //override for dynamic shape plugin
        DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) override;

        int initialize() override;

        virtual void terminate() override {};

        //override the enqueue function
        int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

        virtual size_t getSerializationSize() const override;

        virtual void serialize(void* buffer) const override;

        //override for dynamic shape plugin
        bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override {
            return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
        }

        const char* getPluginType() const override;

        const char* getPluginVersion() const override;

        void destroy() override;

        IPluginV2DynamicExt* clone() const override;

        void setPluginNamespace(const char* pluginNamespace) override;

        const char* getPluginNamespace() const override;

        DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

        void attachToContext(
            cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

        //override for dynamic shape plugin
        void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) override;

        void detachFromContext() override;

    protected:
        using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
        using nvinfer1::IPluginV2DynamicExt::configurePlugin;
        using nvinfer1::IPluginV2DynamicExt::enqueue;
        using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
        using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
        using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
        using nvinfer1::IPluginV2DynamicExt::supportsFormat;

    private:
        void forwardGpu(const float *const * inputs, float * output, cudaStream_t stream, std::vector<int>& kernelDims, int batchSize = 1);
        int mThreadCount = 256;
        const char* mPluginNamespace;
        int mKernelCount;
        int mClassCount;
        int mYoloV5NetWidth;
        int mYoloV5NetHeight;
        int mMaxOutObject;
        std::vector<Yolo::YoloKernel> mYoloKernel;
        void** mAnchor;
    };

    class YoloPluginCreator : public IPluginCreator
    {
    public:
        YoloPluginCreator();

        ~YoloPluginCreator() override = default;

        const char* getPluginName() const override;

        const char* getPluginVersion() const override;

        const PluginFieldCollection* getFieldNames() override;

        IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

        IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

        void setPluginNamespace(const char* libNamespace) override
        {
            mNamespace = libNamespace;
        }

        const char* getPluginNamespace() const override
        {
            return mNamespace.c_str();
        }

    private:
        std::string mNamespace;
        static PluginFieldCollection mFC;
        static std::vector<PluginField> mPluginAttributes;
    };
    REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
};

#endif 
