#ifndef __FACETHINK_API_FRAME_STATE_DETECTOR_HPP__
#define __FACETHINK_API_FRAME_STATE_DETECTOR_HPP__

#include <string>

#ifdef WIN32
#include "paas_uploader.h"
#endif //WIN32

#include "config.hpp"
#include "det_timu_yolov5.hpp"
#include "image_to_blob.hpp"
#include "tensorrt.hpp"

namespace facethink {
	namespace dettimuyolo {
		using namespace tensorrt;
		class FrameStateDetector : public DetTimuYolo {
		public:

			explicit FrameStateDetector(
				const std::string& det_model_file,
				const std::string& config_file);

			virtual ~FrameStateDetector();
			virtual int detection(std::vector<cv::Mat>& input_imgs,
                                  const cv::Mat &srcMat, 
                                  int angle,         
                                  vector<int> &anchor,                                                           
                                  std::vector<std::vector<int>>& final_boxes, 
                                  std::vector<std::vector<int>>& final_show_boxes,
                                  std::vector<int> &anchor_box, 
                                  std::vector<int> &anchor_show_box,  
								  double &predict_used, 
								  double &post_used,                                 
                                  bool is_rgb_format)
		protected:
			int preprocess(std::vector<cv::Mat>& imgs, cv::Mat& det_blob_img, bool is_rgb_format);
			//int detFeature(std::vector<cv::Mat>& input_imgs, bool is_rgb_format = false);
			void coordinatRotation(int largeRotate, std::vector<int> &anchor,const cv::Mat &pic);
			int find_anchor_index(const std::vector<cv::Rect> &rect_vec, const std::vector<int> &anchor);
			int detFeature(std::vector<cv::Mat>& imgs,const cv::Mat &srcMat, int angle, vector<int> &anchor,  std::vector<cv::Rect>& final_boxes, std::vector<cv::Rect>& final_show_boxes, cv::Rect &anchor_rect, cv::Rect &anchor_show_rect, double &predict_used, double &post_used, bool is_rgb_format);

			cv::Mat letterbox(cv::Mat& img);
			void Nms(std::vector<std::vector<float>> bboxes,std::vector<std::vector<float>> &after_nms_bboxes);
			void sort_vector(std::vector<std::vector<float>>& boxes, std::vector<int>& seq);
			float minimum(float f1, float f2);
			float maximum(float f1, float f2);
			double IoU(const std::vector<float>& rect1, const std::vector<float>& rect2);
			//void areaOverlap(const cv::Rect& rectI, const cv::Rect rectJ, vector<float> &ratios);
			//bool comp_in_column(cv::Rect rect_a, cv::Rect rect_b);

			int analysisLayout(const cv::Mat &inMat, 
							std::vector<cv::Rect>rectVec,
							std::vector<cv::Rect>&outRectVec,
							std::vector<std::vector<cv::Rect>> &divideBoxs, 
							string image_name);

			void post_process(const cv::Mat inMat, 
							const string &image_name, 
							std::vector<std::vector<float>> &detect_results, 
							std::vector<cv::Rect> &final_boxes,
							std::vector<cv::Rect> &final_show_boxes);

			void setOriginW(int w) { origin_w = w; }
			void setOriginH(int h) { origin_h = h; }
			int getOriginH() { return origin_h; }
			int getOriginW() { return origin_w; }
			void setInputDims(int batch, int chs, int h, int w) 
			{ 
				input_dims.clear();
				input_dims.push_back(batch);
				input_dims.push_back(chs); 
				input_dims.push_back(h); 
				input_dims.push_back(w);
			}
			std::vector<int> getInputDims() { return input_dims; }
		private:
			Config config_;
			Tensorrt* tensorrt_;
			int max_batch_ = 1;
			int INPUT_W = 512;
			int INPUT_H = 320;
			int origin_w;
			int origin_h;
			std::vector<int> input_dims;

#ifdef WIN32
			analyze::PaasUploader uploader_;
			int sdk_state_; // 0:normal use; -5: offline authentication fail; -6:wrong params of uploading module
#endif //WIN32
		};

	}
}

#endif
