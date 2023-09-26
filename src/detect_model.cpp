#include "automatic_box.h"
#include <time.h>

#include <mutex>
std::mutex det_api_mutex;

/**
 * 此处对HandleRequest包装一层的目的：防止当前HandleRequest不能很好的
 * 满足后续的需求；若能满足，也可以直接在service_main.cpp的Listen中直
 * 接调用HandleRequest方法
 */

std::string AutomaticBox::detect_model_;
std::string AutomaticBox::mask_model_;
std::string AutomaticBox::detect_cfg_;
std::string AutomaticBox::cls_model_;
std::string AutomaticBox::cls_config_;
std::unique_ptr<DetTimuYolo> AutomaticBox::p_detect_ = nullptr;
std::unique_ptr<ClsTopic> AutomaticBox::p_cls_detect_ = nullptr;

bool AutomaticBox::Init() {
    // 检测模型
    detect_model_   = "/home/guoweiye/roration_model/src/alg/yolo_model/yolov5l_324_fp16.engine";
    mask_model_     = "/home/guoweiye/roration_model/src/alg/yolo_model/mask_trt.engine";
    detect_cfg_     = "/home/guoweiye/roration_model/src/alg/yolo_model/config.ini";
    cls_model_      = "/home/guoweiye/roration_model/src/alg/cls_model/tal_2_cls.onnx";
    cls_config_     = "/home/guoweiye/roration_model/src/alg/cls_model/config.ini";

    if (nullptr == p_detect_) {
        p_detect_.reset(DetTimuYolo::create(detect_model_, mask_model_, detect_cfg_));
    }

    if (nullptr == p_cls_detect_) {
        p_cls_detect_.reset(ClsTopic::create(cls_model_, cls_config_));
    }
    return true;
}

bool AutomaticBox::handler(Json::Value &result,cv::Mat &cv_image_) {
    // 处理前系统时间
    time_t process_start_time;
    time(&process_start_time);

    // TODO: 业务处理及调用模型等，下面是模拟输出结果
    std::vector<cv::Mat> det_imgs;
    det_imgs.emplace_back(cv_image_);
    int angle_index = 0;

    // DetApi
    // 识别框
    int err_code = -1;
    std::vector<int> anchor_boxes;
    std::vector<std::vector<int>> final_boxes;
    cv::Rect anchor_rect, anchor_show_rect;
    std::vector<cv::Rect> final_rects, final_show_rects;
    {
        std::unique_lock<std::mutex> lock(det_api_mutex);
        err_code = p_detect_->detection(det_imgs, cv_image_,
                                        final_boxes,
                                        angle_index,
                                        anchor_, anchor_boxes,
                                        final_rects, final_show_rects,
                                        anchor_rect, anchor_show_rect,
                                        false);
        // 处理结束时间
        time_t process_end_time;
        time(&process_end_time);

        // 处理时间
        result["process_time"] = difftime(process_start_time,process_end_time);
    }

    if (0 != err_code) {
        return false;
    }

    // 每个题目的类型 0表示非计算题 1表示计算题, type在request json中赋值,未赋值的默认为0
    std::vector<int> cls_types;
    if (1 == type_) {
        for (int i = 0; i < final_boxes.size(); i++) {
            cv::Rect roi    = final_rects[i];
            cv::Mat img_cut = det_imgs[0](roi);
            // 调用题型分类模型
            std::vector<std::vector<float>> cls_probs;
            p_cls_detect_->detection({img_cut}, cls_probs);
            int type = cls_probs[0][1] > cls_probs[0][0] ? 1 : 0;
            cls_types.push_back(type);
        }
    }
    
    result["data"]   = Json::arrayValue;
    result["single"] = Json::arrayValue;
    if (0 == err_code) {
        
        result["rotate"] = angle_index;
        
        for (unsigned int i = 0; i < final_rects.size(); ++i) {
            Json::Value data_item;
            
            if (1 == type_)
                data_item["item_type"] = cls_types[i];
            
            data_item["item_level"] = Json::nullValue;
            
            cv::Rect img_rect = final_rects[i];
            Json::Value item_pos_json;
            ProcessRect(img_rect, item_pos_json);
            data_item["item_position"] = item_pos_json["data"];
            
            cv::Rect img_rect_show = final_show_rects[i];
            Json::Value item_show_json;
            ProcessRect(img_rect_show, item_show_json);
            data_item["item_position_show"] = item_show_json["data"];
            
            for (unsigned int j = 0; j < final_boxes[i].size();) {
                Json::Value item_pos = Json::arrayValue;
                item_pos.append(final_boxes[i][j]);
                item_pos.append(final_boxes[i][j+1]);
                data_item["item_position_rotate"].append(item_pos);
                
                j += 2;
            }
            
            result["data"].append(data_item);
        }

        Json::Value single_json;
        Json::Value single_pos_json;
        ProcessRect(anchor_rect, single_pos_json);
        single_json["item_position"] = single_pos_json["data"];

        Json::Value single_show_json;
        ProcessRect(anchor_show_rect, single_show_json);
        single_json["item_position_show"] = single_show_json["data"];

        for (unsigned int i = 0; i < anchor_boxes.size();) {
            Json::Value item_pos = Json::arrayValue;
            item_pos.append(anchor_boxes[i]);
            item_pos.append(anchor_boxes[i+1]);
            single_json["item_position_rotate"].append(item_pos);
            
            i += 2;
        }
        if (anchor_boxes.size() > 0)
            result["single"].append(single_json);
    }

    return true;
}

void AutomaticBox::ProcessRect(const cv::Rect &rect, Json::Value &result) {
    // 四点位置
    Json::Value item_pos1 = Json::arrayValue;
    item_pos1.append(rect.x);
    item_pos1.append(rect.y);
    result["data"].append(item_pos1);
    
    Json::Value item_pos2 = Json::arrayValue;
    item_pos2.append(rect.x + rect.width);
    item_pos2.append(rect.y);
    result["data"].append(item_pos2);
    
    Json::Value item_pos3 = Json::arrayValue;
    item_pos3.append(rect.x + rect.width);
    item_pos3.append(rect.y + rect.height);
    result["data"].append(item_pos3);
    
    Json::Value item_pos4 = Json::arrayValue;
    item_pos4.append(rect.x);
    item_pos4.append(rect.y + rect.height);
    result["data"].append(item_pos4);
}

std::string getLevelStr(int level)
{
	std::string levelStr = "";
	for (int i = 0; i < level; i++)
	{
		levelStr += "\t"; //这里可以\t换成你所需要缩进的空格数
	}
	return levelStr;

}

std::string formatJson(std::string json)
{
	std::string result = "";
	int level = 0;
	for (std::string::size_type index = 0; index < json.size(); index++)
	{
		char c = json[index];

		if (level > 0 && '\n' == json[json.size() - 1])
		{
			result += getLevelStr(level);
		}

		switch (c)
		{
		case '{':
		case '[':
			result = result + c + "\n";
			level++;
			result += getLevelStr(level);
			break;
		case ',':
			result = result + c + "\n";
			result += getLevelStr(level);
			break;
		case '}':
		case ']':
			result += "\n";
			level--;
			result += getLevelStr(level);
			result += c;
			break;
		default:
			result += c;
			break;
		}

	}
	return result;
}

int main()
{
    AutomaticBox dect;
    Json::Value result;
    cv::Mat image = cv::imread("./image.jpeg");

    if(dect.Init()){
        std::cout<<"init dect_model successfully"<<std::endl;
    }
    else{
        std::cout<<"init model failed!"<<std::endl;
        return -1;
    }

    if(dect.handler(result,image)){
        std::cout<<"dect successfully"<<std::endl;
    }
    else{
        std::cout<<"dect failed"<<std::endl;
        return -1;
    }
    
    // 输出Json数据
    std::cout<<formatJson(result.toStyledString())<<std::endl;

    return 0;
}
