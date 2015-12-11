/*
 *  Copyright 2011-2014 Maxim Milakov
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "image_classifier_demo_toolset.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/thread/thread.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

#include <exception>

const unsigned int image_classifier_demo_toolset::top_n = 5;
const char * image_classifier_demo_toolset::class_names_filename = "class_names.txt";
const int image_classifier_demo_toolset::border = 5;
const int image_classifier_demo_toolset::border_text = 2;
const int image_classifier_demo_toolset::font_face = cv::FONT_HERSHEY_SIMPLEX;
const float image_classifier_demo_toolset::prob_part = 0.25F;

image_classifier_demo_toolset::image_classifier_demo_toolset(nnforge::factory_generator::ptr factory)
	: nnforge::toolset(factory)
	, font_scale(0.4)
	, text_thickness(1)
{
}

image_classifier_demo_toolset::~image_classifier_demo_toolset()
{
}

std::string image_classifier_demo_toolset::get_default_action() const
{
	return "demo";
}

void image_classifier_demo_toolset::do_custom_action()
{
	if (!action.compare("demo"))
	{
		run_demo();
	}
	else
	{
		toolset::do_custom_action();
	}
}

void image_classifier_demo_toolset::dump_help() const
{
	std::cout << "KEYS:" << std::endl;
	std::cout << "q,Esc : quit application" << std::endl;
	std::cout << "+/- : increase/decrease font size" << std::endl;
	std::cout << "s : save screenshot into screenshots directory" << std::endl;
}

void image_classifier_demo_toolset::run_demo()
{
	cv::VideoCapture cap(0);
	if (!cap.isOpened())
		throw std::runtime_error("Unable to open capture device (camera/video)");

	load_cls_class_info();

	init_input_config();
	std::cout << "Network expects input image of size " << input_config.dimension_sizes[0] << "x" <<  input_config.dimension_sizes[1] << std::endl;

	normalizer = get_normalize_data_transformer(input_layer_name);

	dump_help();

	error_message.clear();
	safe_set_fps(-1.0F);
	last_report_time = boost::chrono::high_resolution_clock::now();
	safe_set_demo_should_stop(false);
	boost::thread classifier_thread(callable(*this));
	try
	{
		init_draw_params();

		std::string window_name = "ImageNet Demo";
		cv::namedWindow(window_name);
		cv::Mat frame;
		while (!safe_peek_demo_should_stop())
		{
			cap >> frame;
			if (frame.empty())
				break;

			set_input_data(frame);

			add_classifier_results(frame);

			cv::imshow(window_name, frame);

			if (should_report_stats)
				report_stats();

			char key = (char)cv::waitKey(5);
			switch (key) {
				case 'q':
				case 'Q':
				case 27: //escape key
					safe_set_demo_should_stop(true);
					break;
				case '+':
					font_scale = std::min(font_scale + 0.2, 1.2);
					init_draw_params();
					break;
				case '-':
					font_scale = std::max(font_scale - 0.2, 0.4);
					init_draw_params();
					break;
				case 's':
				case 'S':
					save_image(frame);
					break;
				default:
					break;
			}
		}

		safe_set_demo_should_stop(true);
		classifier_thread.join();
	}
	catch (std::exception&)
	{
		safe_set_demo_should_stop(true);
		classifier_thread.join();
		throw;
	}

	if (!error_message.empty())
		throw std::runtime_error(error_message);
}

void image_classifier_demo_toolset::init_draw_params()
{
	text_thickness = std::max(1, static_cast<int>(font_scale * 3));
	text_prob_part = std::min(1.0F - prob_part, static_cast<float>(font_scale) * 0.6F);

	{
		int baseline = 0;
		cv::Size text_size = cv::getTextSize(
			"FPS",
			font_face,
			font_scale,
			text_thickness,
			&baseline);
		text_height = text_size.height + text_thickness;
	}
}

bool compare_results(std::pair<unsigned int, float> elem1, std::pair<unsigned int, float> elem2)
{
	if (elem1.second > elem2.second)
		return true;
	else if (elem1.second < elem2.second)
		return false;
	else
		return (elem1.first < elem2.first);
}

void image_classifier_demo_toolset::run_classifier_loop()
{
	try
	{
		nnforge::network_schema::ptr schema = load_schema();
		nnforge::forward_propagation::ptr forward_prop = forward_prop_factory->create(*schema, inference_output_layer_names, debug);
		{
			nnforge::network_data data;
			boost::filesystem::path data_path = get_working_data_folder() / ann_subfolder_name / (boost::format("ann_trained_%|1$03d|") % 0).str();
			data.read(data_path);
			forward_prop->set_data(data);
		}
		std::map<std::string, nnforge::layer_configuration_specific> layer_config_map;
		layer_config_map.insert(std::make_pair(input_layer_name, input_config));
		forward_prop->set_input_configuration_specific(layer_config_map);

		last_write = boost::chrono::high_resolution_clock::now();
		forward_prop->run(*this, *this);
	}
	catch(std::exception& e)
	{
		safe_set_demo_should_stop(true);
		error_message = e.what();
	}
}

void image_classifier_demo_toolset::report_stats()
{
	boost::chrono::steady_clock::time_point new_report_time = boost::chrono::high_resolution_clock::now();
	boost::chrono::duration<float> run_duration = new_report_time - last_report_time;
	std::cout << "Frame FPS=" << (1.0F / run_duration.count());
	last_report_time = new_report_time;

	float current_fps = safe_peek_fps();
	if (current_fps > 0.0F)
		std::cout << ", Classifier FPS=" << current_fps;

	nnforge_shared_ptr<std::vector<std::pair<unsigned int, float> > > output_data = safe_peek_output_data();
	if (output_data)
	{
		for(int i = 0; i < output_data->size(); ++i)
		{
			const std::pair<unsigned int, float>& current_elem = output_data->at(i);
			std::cout << ", " << get_class_name_by_class_id(current_elem.first) << " " << current_elem.second;
		}
	}

	std::cout << std::endl;
}

void image_classifier_demo_toolset::add_classifier_results(cv::Mat3b frame)
{
	float current_fps = safe_peek_fps();
	if (current_fps > 0.0F)
		add_classifier_fps(frame, current_fps);

	nnforge_shared_ptr<std::vector<std::pair<unsigned int, float> > > output_data = safe_peek_output_data();
	if (output_data && (!output_data->empty()))
		add_classifier_output(frame, *output_data);
}

void image_classifier_demo_toolset::add_classifier_output(cv::Mat3b frame, const std::vector<std::pair<unsigned int, float> >& output_data)
{
	int current_bottom = frame.rows;
	for(std::vector<std::pair<unsigned int, float> >::const_reverse_iterator it = output_data.rbegin(); it != output_data.rend(); ++it)
	{
		const std::pair<unsigned int, float>& current_elem = *it;

		{
			cv::Mat3b sub_image = frame.rowRange(current_bottom - (border + text_height + border_text * 2), current_bottom - border).colRange(static_cast<int>(frame.cols * (1.0F - text_prob_part)) + border / 2, frame.cols - border);
			sub_image.convertTo(sub_image, -1, 1.0, 100.0);
			cv::putText(
				sub_image,
				get_class_name_by_class_id(current_elem.first),
				cv::Point(border_text, border_text + text_height - text_thickness),
				font_face,
				font_scale,
				0,
				text_thickness);
		}

		{
			cv::Mat3b sub_image = frame.rowRange(current_bottom - (border + text_height + border_text * 2), current_bottom - border).colRange(border, static_cast<int>(frame.cols * prob_part) - border / 2);
			sub_image.convertTo(sub_image, -1, 1.0, 100.0);
			int total_width = sub_image.cols - (border_text * 2);
			int bar_width = static_cast<int>(total_width * current_elem.second);
			cv::Mat3b sub_image2 = sub_image.rowRange(border_text, text_height + border_text).colRange(sub_image.cols - (border_text + bar_width), sub_image.cols - border_text);
			sub_image2.convertTo(sub_image2, -1, 1.0, -150.0);
		}

		current_bottom -= border + text_height + border_text * 2;
	}
}

void image_classifier_demo_toolset::add_classifier_fps(cv::Mat3b frame, float fps)
{
	std::string text = (boost::format("FPS: %|1$.1f|") % fps).str();

	cv::Mat3b sub_image = frame.rowRange(border, border + text_height + border_text * 2).colRange(border, frame.cols - border / 2);

	sub_image.convertTo(sub_image, -1, 1.0, 100.0);

	cv::putText(
		sub_image,
		text,
		cv::Point(border_text, border_text + text_height - text_thickness),
		font_face,
		font_scale,
		0,
		text_thickness);
}

nnforge_shared_ptr<std::vector<float> > image_classifier_demo_toolset::safe_peek_input_data()
{
	boost::lock_guard<boost::mutex> guard(input_data_mutex);

	return input_data_smart_ptr;
}

void image_classifier_demo_toolset::safe_set_input_data(nnforge_shared_ptr<std::vector<float> > val)
{
	boost::lock_guard<boost::mutex> guard(input_data_mutex);

	input_data_smart_ptr = val;
}

nnforge_shared_ptr<std::vector<std::pair<unsigned int, float> > > image_classifier_demo_toolset::safe_peek_output_data()
{
	boost::lock_guard<boost::mutex> guard(output_data_mutex);

	return output_data_smart_ptr;
}

void image_classifier_demo_toolset::safe_set_output_data(nnforge_shared_ptr<std::vector<std::pair<unsigned int, float> > > val)
{
	boost::lock_guard<boost::mutex> guard(output_data_mutex);

	output_data_smart_ptr = val;
}

bool image_classifier_demo_toolset::safe_peek_demo_should_stop()
{
	boost::lock_guard<boost::mutex> guard(demo_should_stop_mutex);

	return demo_should_stop;
}

void image_classifier_demo_toolset::safe_set_demo_should_stop(bool val)
{
	boost::lock_guard<boost::mutex> guard(demo_should_stop_mutex);

	demo_should_stop = val;
}

float image_classifier_demo_toolset::safe_peek_fps()
{
	boost::lock_guard<boost::mutex> guard(fps_mutex);

	return fps;
}

void image_classifier_demo_toolset::safe_set_fps(float val)
{
	boost::lock_guard<boost::mutex> guard(fps_mutex);

	fps = val;
}

void image_classifier_demo_toolset::init_input_config()
{
	nnforge::network_schema::ptr schema = load_schema();

	if (inference_output_layer_names.size() != 1)
		throw nnforge::neural_network_exception("You shouls set inference_output_layer_name exactly once");

	std::map<std::string, nnforge::layer_configuration_specific> output_configuration_specific_map;
	output_configuration_specific_map.insert(std::make_pair(inference_output_layer_names[0], nnforge::layer_configuration_specific(static_cast<unsigned int>(class_id_to_class_name_map.size()), std::vector<unsigned int>(2, 1))));
	std::map<std::string, nnforge::layer_configuration_specific> input_configuration_specific_map = schema->get_layer_configuration_specific_map_reverse(output_configuration_specific_map);

	std::map<std::string, nnforge::layer_configuration_specific>::const_iterator it = input_configuration_specific_map.find(input_layer_name);
	if (it == input_configuration_specific_map.end())
		throw nnforge::neural_network_exception((boost::format("Cannot determine config for layer %1%") % input_layer_name).str());

	input_config = it->second;
}

void image_classifier_demo_toolset::set_input_data(cv::Mat original_image, bool truncate_image)
{
	cv::Mat3b dest_image;
	if (truncate_image)
	{
		float width_ratio = static_cast<float>(original_image.cols) / static_cast<float>(input_config.dimension_sizes[0]);
		float height_ratio = static_cast<float>(original_image.rows) / static_cast<float>(input_config.dimension_sizes[1]);
		cv::Mat3b source_sub_image;
		if (width_ratio > height_ratio)
		{
			unsigned int source_sub_image_width = static_cast<unsigned int>(input_config.dimension_sizes[0] * height_ratio + 0.5F);
			unsigned int start_col = (original_image.cols - source_sub_image_width) / 2;
			source_sub_image = original_image.colRange(start_col, start_col + source_sub_image_width);
		}
		else
		{
			unsigned int source_sub_image_height = static_cast<unsigned int>(input_config.dimension_sizes[1] * width_ratio + 0.5F);
			unsigned int start_row = (original_image.rows - source_sub_image_height) / 2;
			source_sub_image = original_image.rowRange(start_row, start_row + source_sub_image_height);
		}
		cv::resize(source_sub_image, dest_image, cv::Size(input_config.dimension_sizes[0], input_config.dimension_sizes[1]), 0.0, 0.0);
	}
	else
	{
		dest_image = cv::Mat3b(input_config.dimension_sizes[1], input_config.dimension_sizes[0], cv::Vec3b(128, 128, 128));
		float width_ratio = static_cast<float>(input_config.dimension_sizes[0]) / static_cast<float>(original_image.cols);
		float height_ratio = static_cast<float>(input_config.dimension_sizes[1]) / static_cast<float>(original_image.rows);
		cv::Mat3b dest_sub_image;
		if (width_ratio > height_ratio)
		{
			unsigned int dest_sub_image_width = static_cast<unsigned int>(original_image.cols * height_ratio + 0.5F);
			unsigned int start_col = (dest_image.cols - dest_sub_image_width) / 2;
			dest_sub_image = dest_image.colRange(start_col, start_col + dest_sub_image_width);
		}
		else
		{
			unsigned int dest_sub_image_height = static_cast<unsigned int>(original_image.rows * width_ratio + 0.5F);
			unsigned int start_row = (dest_image.rows - dest_sub_image_height) / 2;
			dest_sub_image = dest_image.rowRange(start_row, start_row + dest_sub_image_height);
		}
		cv::resize(original_image, dest_sub_image, cv::Size(dest_sub_image.cols, dest_sub_image.rows), 0.0, 0.0);
	}

	nnforge_shared_ptr<std::vector<float> > new_input_data(new std::vector<float>(dest_image.rows * dest_image.cols * 3));

	float * input_data = &(*new_input_data->begin());

	float * r_dst_it = input_data;
	float * g_dst_it = input_data + (dest_image.rows * dest_image.cols);
	float * b_dst_it = input_data + (dest_image.rows * dest_image.cols * 2);
	for(cv::Mat3b::const_iterator it = dest_image.begin(); it != dest_image.end(); ++it, ++r_dst_it, ++g_dst_it, ++b_dst_it)
	{
		*r_dst_it = static_cast<float>((*it)[2]) * (1.0F / 255.0F);
		*g_dst_it = static_cast<float>((*it)[1]) * (1.0F / 255.0F);
		*b_dst_it = static_cast<float>((*it)[0]) * (1.0F / 255.0F);
	}

	if (normalizer)
		normalizer->transform(input_data, input_data, input_config, 0);

	safe_set_input_data(new_input_data);
}

std::vector<nnforge::bool_option> image_classifier_demo_toolset::get_bool_options()
{
	std::vector<nnforge::bool_option> res = toolset::get_bool_options();

	res.push_back(nnforge::bool_option("report_stats", &should_report_stats, false, "Report stats to console"));

	return res;
}

std::vector<nnforge::string_option> image_classifier_demo_toolset::get_string_options()
{
	std::vector<nnforge::string_option> res = toolset::get_string_options();

	res.push_back(nnforge::string_option("input_layer_name", &input_layer_name, "images", "Input layer name"));

	return res;
}

void image_classifier_demo_toolset::load_cls_class_info()
{
	class_id_to_class_name_map.clear();

	boost::filesystem::path cls_class_info_filepath = get_working_data_folder() / class_names_filename;

	std::cout << "Reading class names from " << cls_class_info_filepath.string() << "...";

	boost::filesystem::ifstream file_input(cls_class_info_filepath, std::ios_base::in);

	std::string str;
	std::getline(file_input, str); // Skip header
	int line_number = 1;
	while (true)
	{
		++line_number;
		std::getline(file_input, str);
		if (str.empty())
			break;
		std::vector<std::string> strs;
		boost::split(strs, str, boost::is_any_of("\t"));

		if (strs.size() != 2)
			throw std::runtime_error((boost::format("Wrong number of fields in line %1%: %2%") % line_number % str).str());

		int class_id = atol(strs[0].c_str());
		std::string class_name = strs[1];
		boost::trim(class_name);

		class_id_to_class_name_map.insert(std::make_pair(class_id, class_name));
	}
	std::cout << "done" << std::endl;
	std::cout << class_id_to_class_name_map.size() << " class names read" << std::endl;
}

std::string image_classifier_demo_toolset::get_class_name_by_class_id(unsigned int class_id) const
{
	std::map<unsigned int, std::string>::const_iterator it = class_id_to_class_name_map.find(class_id);
	if (it == class_id_to_class_name_map.end())
		return (boost::format("Unknown class %1%") % class_id).str();

	return it->second;
}

void image_classifier_demo_toolset::save_image(cv::Mat3b frame)
{
	boost::filesystem::path screenshots_folder = get_working_data_folder() / "screenshots";
	boost::filesystem::create_directories(screenshots_folder);

	std::string current_datetime;
	{
		time_t rawtime;
		struct tm * timeinfo;
		time(&rawtime);
		timeinfo = localtime(&rawtime);
		char buf[80];
		strftime(buf, 80, "%Y-%m-%d_%H%M%S", timeinfo);
		current_datetime = buf;
	}

	std::string filename = (boost::format("image_classifier_screenshot_%1%.jpg") % current_datetime).str();
	boost::filesystem::path screenshot_filepath = screenshots_folder / filename;
	cv::imwrite(screenshot_filepath.string(), frame);
}

std::map<std::string, nnforge::layer_configuration_specific> image_classifier_demo_toolset::get_config_map() const
{
	std::map<std::string, nnforge::layer_configuration_specific> res;
	res.insert(std::make_pair(input_layer_name, input_config));
	return res;
}

bool image_classifier_demo_toolset::read(
	unsigned int entry_id,
	const std::map<std::string, float *>& data_map)
{
	if (safe_peek_demo_should_stop())
		return false;

	nnforge_shared_ptr<std::vector<float> > input_data = safe_peek_input_data();
	while (!input_data)
	{
		boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
		input_data = safe_peek_input_data();
		if (safe_peek_demo_should_stop())
			return false;
	}

	memcpy(data_map.find(input_layer_name)->second, &input_data->at(0), input_data->size() * sizeof(float));

	return true;
}

void image_classifier_demo_toolset::set_epoch(unsigned int epoch_id)
{
}

int image_classifier_demo_toolset::get_entry_count() const
{
	return 1;
}

void image_classifier_demo_toolset::set_config_map(const std::map<std::string, nnforge::layer_configuration_specific> config_map)
{
}

void image_classifier_demo_toolset::write(const std::map<std::string, const float *>& data_map)
{
	boost::chrono::steady_clock::time_point new_last_write = boost::chrono::high_resolution_clock::now();
	boost::chrono::duration<float> run_duration = new_last_write - last_write;
	last_write = new_last_write;
	safe_set_fps(1.0F / run_duration.count());

	const float * data = data_map.find(inference_output_layer_names[0])->second;
	std::vector<std::pair<unsigned int, float> > full_output(class_id_to_class_name_map.size());
	for(unsigned int class_id = 0; class_id < full_output.size(); ++class_id)
		full_output[class_id] = std::make_pair(class_id, data[class_id]);

	unsigned int top_n_actual = std::min(top_n, (unsigned int)full_output.size());
	std::partial_sort(full_output.begin(), full_output.begin() + top_n_actual, full_output.end(), compare_results);

	nnforge_shared_ptr<std::vector<std::pair<unsigned int, float> > > new_output(new std::vector<std::pair<unsigned int, float> >(top_n_actual));
	for(unsigned int i = 0; i < top_n_actual; ++i)
		new_output->at(i) = full_output[i];
	safe_set_output_data(new_output);
}
