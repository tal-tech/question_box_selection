#ifndef __FACETHINK_API_TIMU_YOLO_CONFIG_HPP__
#define __FACETHINK_API_TIMU_YOLO_CONFIG_HPP__

#include <vector>
#include <string>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

namespace facethink {
	typedef boost::log::sinks::synchronous_sink< boost::log::sinks::text_file_backend > sink_t;
  	namespace dettimuyolo {
		class Config {
		public:
			Config() {

				boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::error);
			}

		public:
			void ReadIniFile(const std::string& config_file); // class Config
			void StopFileLogging();

			int max_batch_ = 1;
			int input_w = 512;
			int input_h = 512;


#ifdef WIN32
			// params of offline authentication
			std::string auth_app_key_;
			std::string auth_secret_key_;

			// params of uploading data
			std::string access_key_id_;
			std::string access_key_secret_;
			std::string server_url_;
			std::string business_id_;
			std::string business_name_;
			std::string business_key_;
			std::string api_id_;
			std::string api_name_;
			std::string sdk_version_;
			std::string app_key_;
			int batch_count_;
			int batch_interval_;
			// types of uploading data;0:do not upload;1:only upload count
			int upload_type_;
#endif //WIN32
		private:
			boost::shared_ptr< sink_t > file_sink;
		};

	}	
}

#endif
