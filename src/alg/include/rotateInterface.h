#include <iostream>
#include <fstream>
#include "cls_image_rotate.hpp"
#include <chrono>
#include <thread>
#include <sys/timeb.h>
#include<map>
#pragma once

namespace facethink
{
    class ClsImageRotate;
}

class jietiRotateModel
{
public:
    jietiRotateModel();
    virtual ~jietiRotateModel();
    int Init(const std::string& model_file, const std::string& config_file);
    facethink::ClsImageRotate* getModel();
private:
    facethink::ClsImageRotate* m_model;
};



class rotateInterface
{
  public:
    rotateInterface();
    // int Init(string rotate_model,string config_path);
    int getAngle(jietiRotateModel *rotate_model,cv::Mat &img_ori,int &angelIndex,std::vector<float>& probs);
    int getHWClass(jietiRotateModel *class_model,const cv::Mat &img_ori,bool &type);
    // private:
    //     ClsImageRotate *rotate_model; 
};
