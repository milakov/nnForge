/*
 *  Copyright 2011-2015 Maxim Milakov
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

#include "gtsrb_toolset.h"

#include <random>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/algorithm/string.hpp>
#include <algorithm>
#include <boost/filesystem/fstream.hpp>
#include <boost/format.hpp>
#include <iostream>

const unsigned int gtsrb_toolset::image_width = 32;
const unsigned int gtsrb_toolset::image_height = 32;
const unsigned int gtsrb_toolset::class_count = 43;
const bool gtsrb_toolset::use_roi = true;
const float gtsrb_toolset::max_rotation_angle_in_degrees = 15.0F;
const float gtsrb_toolset::max_scale_factor = 1.1F;
const float gtsrb_toolset::max_shift = 2.0F;
const float gtsrb_toolset::max_contrast_factor = 1.5F;
const float gtsrb_toolset::max_brightness_shift = 0.2F;

gtsrb_toolset::gtsrb_toolset(nnforge::factory_generator::ptr factory)
	: nnforge::toolset(factory)
{
}

gtsrb_toolset::~gtsrb_toolset()
{
}

void gtsrb_toolset::prepare_training_data()
{
	{
		boost::filesystem::path image_file_path = get_working_data_folder() / "training_images.dt";
		boost::filesystem::path label_file_path = get_working_data_folder() / "training_labels.dt";
		std::cout << "Writing data to " << image_file_path.string() << " and " << label_file_path.string() << std::endl;

		std::shared_ptr<std::ofstream> image_file_stream(new boost::filesystem::ofstream(image_file_path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc));
		nnforge::layer_configuration_specific input_configuration;
		input_configuration.feature_map_count = 1;
		input_configuration.dimension_sizes.push_back(image_width);
		input_configuration.dimension_sizes.push_back(image_height);
		nnforge::structured_data_stream_writer image_writer(
			image_file_stream,
			input_configuration);

		std::shared_ptr<std::ofstream> label_file_stream(new boost::filesystem::ofstream(label_file_path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc));
		nnforge::layer_configuration_specific output_configuration;
		output_configuration.feature_map_count = class_count;
		output_configuration.dimension_sizes.push_back(1);
		output_configuration.dimension_sizes.push_back(1);
		nnforge::structured_data_stream_writer label_writer(
			label_file_stream,
			output_configuration);

		for(unsigned int folder_id = 0; folder_id < class_count; ++folder_id)
		{
			boost::filesystem::path subfolder_name = boost::filesystem::path("Final_Training") / "Images" / (boost::format("%|1$05d|") % folder_id).str();
			std::string annotation_file_name = (boost::format("GT-%|1$05d|.csv") % folder_id).str();

			write_folder(
				image_writer,
				label_writer,
				subfolder_name,
				annotation_file_name.c_str());
		}
	}
	
	{
		boost::filesystem::path image_file_path = get_working_data_folder() / "validating_images.dt";
		boost::filesystem::path label_file_path = get_working_data_folder() / "validating_labels.dt";
		std::cout << "Writing data to " << image_file_path.string() << " and " << label_file_path.string() << std::endl;

		std::shared_ptr<std::ofstream> image_file_stream(new boost::filesystem::ofstream(image_file_path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc));
		nnforge::layer_configuration_specific input_configuration;
		input_configuration.feature_map_count = 1;
		input_configuration.dimension_sizes.push_back(image_width);
		input_configuration.dimension_sizes.push_back(image_height);
		nnforge::structured_data_stream_writer image_writer(
			image_file_stream,
			input_configuration);

		std::shared_ptr<std::ofstream> label_file_stream(new boost::filesystem::ofstream(label_file_path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc));
		nnforge::layer_configuration_specific output_configuration;
		output_configuration.feature_map_count = class_count;
		output_configuration.dimension_sizes.push_back(1);
		output_configuration.dimension_sizes.push_back(1);
		nnforge::structured_data_stream_writer label_writer(
			label_file_stream,
			output_configuration);

		boost::filesystem::path subfolder_name = boost::filesystem::path("Final_Test") / "Images";
		std::string annotation_file_name = "GT-final_test.csv";

		write_folder(
			image_writer,
			label_writer,
			subfolder_name,
			annotation_file_name.c_str());
	}
}

void gtsrb_toolset::write_folder(
	nnforge::structured_data_stream_writer& image_writer,
	nnforge::structured_data_stream_writer& label_writer,
	const boost::filesystem::path& relative_subfolder_path,
	const char * annotation_file_name)
{
	boost::filesystem::path subfolder_path = get_input_data_folder() / relative_subfolder_path;
	boost::filesystem::path annotation_file_path = subfolder_path / annotation_file_name;

	std::cout << "Reading input data from " << subfolder_path.string() << std::endl;

	boost::filesystem::ifstream file_input(annotation_file_path, std::ios_base::in);

	nnforge::random_generator generator = nnforge::rnd::get_random_generator();
	std::uniform_real_distribution<float> rotate_angle_distribution(-max_rotation_angle_in_degrees, max_rotation_angle_in_degrees);
	std::uniform_real_distribution<float> scale_distribution(1.0F / max_scale_factor, max_scale_factor);
	std::uniform_real_distribution<float> shift_distribution(-max_shift, max_shift);
	std::uniform_real_distribution<float> contrast_distribution(1.0F / max_contrast_factor, max_contrast_factor);
	std::uniform_real_distribution<float> brightness_shift_distribution(-max_brightness_shift, max_brightness_shift);

	std::string str;
	std::getline(file_input, str); // read the header
	unsigned int entry_read_count = 0;
	while (true)
	{
		std::getline(file_input, str);

		std::vector<std::string> strs;
		boost::split(strs, str, boost::is_any_of(";"));

		if (strs.size() != 8)
			break;

		++entry_read_count;

		std::string file_name = strs[0];
		boost::filesystem::path absolute_file_path = subfolder_path / file_name;

		char* end;
		unsigned int top_left_x = static_cast<unsigned int>(strtol(strs[3].c_str(), &end, 10));
		unsigned int top_left_y = static_cast<unsigned int>(strtol(strs[4].c_str(), &end, 10));
		unsigned int bottom_right_x = static_cast<unsigned int>(strtol(strs[5].c_str(), &end, 10));
		unsigned int bottom_right_y = static_cast<unsigned int>(strtol(strs[6].c_str(), &end, 10));
		unsigned int class_id = static_cast<unsigned int>(strtol(strs[7].c_str(), &end, 10));

		write_single_entry(
			image_writer,
			label_writer,
			absolute_file_path,
			class_id,
			top_left_x,
			top_left_y,
			bottom_right_x,
			bottom_right_y);
	}

	if (entry_read_count == 0)
		throw std::runtime_error((boost::format("No entries with class ID encountered in %1%") % annotation_file_path.string()).str());
}

std::vector<nnforge::data_transformer::ptr> gtsrb_toolset::get_data_transformer_list(
	const std::string& dataset_name,
	const std::string& layer_name,
	dataset_usage usage) const
{
	std::vector<nnforge::data_transformer::ptr> res;

	if ((layer_name == "images") && (dataset_name == "training") && (usage != dataset_usage_check_gradient))
	{
		res.push_back(nnforge::data_transformer::ptr(new nnforge::distort_2d_data_transformer(
			max_rotation_angle_in_degrees,
			max_scale_factor,
			-max_shift,
			max_shift,
			-max_shift,
			max_shift,
			false,
			false,
			1.0F,
			std::numeric_limits<float>::max())));
		res.push_back(nnforge::data_transformer::ptr(new nnforge::intensity_2d_data_transformer(
			max_contrast_factor,
			max_brightness_shift)));
	}

	return res;
}

void gtsrb_toolset::write_single_entry(
		nnforge::structured_data_stream_writer& image_writer,
		nnforge::structured_data_stream_writer& label_writer,
		const boost::filesystem::path& absolute_file_path,
		unsigned int class_id,
		unsigned int roi_top_left_x,
		unsigned int roi_top_left_y,
		unsigned int roi_bottom_right_x,
		unsigned int roi_bottom_right_y)
{
	{
		std::vector<float> inp(image_width * image_height);

		cv::Mat3b image = cv::imread(absolute_file_path.string());

		if (use_roi)
			image = image.rowRange(roi_top_left_y, roi_bottom_right_y).colRange(roi_top_left_x, roi_bottom_right_x);

		cv::Mat3b image_resized;
		cv::resize(image, image_resized, cv::Size(image_width, image_height));

		{
			cv::Mat1b image_monochrome;
			cv::cvtColor(image_resized, image_monochrome, CV_BGR2GRAY);

			std::copy(
				image_monochrome.begin(),
				image_monochrome.end(),
				inp.begin());

			std::vector<float>::iterator dst_it = inp.begin();
			for(cv::Mat1b::const_iterator it = image_monochrome.begin(); it != image_monochrome.end(); ++it, ++dst_it)
				*dst_it = static_cast<float>(*it) * (1.0F / 255.0F);
		}

		image_writer.write(&inp[0]);
	}

	{
		std::vector<float> output(class_count, -1.0F);
		output[class_id] = 1.0F;
		label_writer.write(&output[0]);
	}
}
