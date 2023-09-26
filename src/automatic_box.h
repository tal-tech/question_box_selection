#pragma once
#include "det_timu_yolov5.hpp"
#include "cls_image_rotate_tal.hpp"
#include "./3rdParty/json/include/json.h"
using namespace facethink;

#include <string>
#include <vector>


class AutomaticBox final {
private:
    int                                      rotate_{1};
    int                                      type_{0};
    std::vector<int>                         anchor_;  // 0-x; 1-y
    static std::string                       detect_model_;
    static std::string                       mask_model_;
    static std::string                       detect_cfg_;
    static std::string                       cls_model_;
    static std::string                       cls_config_;
    static std::unique_ptr<DetTimuYolo>      p_detect_;
    static std::unique_ptr<ClsTopic>         p_cls_detect_;

public:
    AutomaticBox(){}

public:
    static bool Init();

public:
    bool handler(Json::Value &result,cv::Mat &image);

private:
    void ProcessRect(const cv::Rect &rect, Json::Value &result);
};
