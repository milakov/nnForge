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

#include "galaxy_zoo_toolset.h"

#include "galaxy_zoo_testing_complete_result_set_visualizer.h"

#include <algorithm>

#include <boost/filesystem/fstream.hpp>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <regex>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

const unsigned int galaxy_zoo_toolset::output_neuron_count = 37;
const float galaxy_zoo_toolset::reserved_for_validation = 0.0F;

const char * galaxy_zoo_toolset::input_training_folder_name = "images_training_rev1";
const char * galaxy_zoo_toolset::input_testing_folder_name = "images_test_rev1";
const char * galaxy_zoo_toolset::training_filename = "training_solutions_rev1.csv";
const char * galaxy_zoo_toolset::testing_rec_ids_filename = "testing_rec_ids.txt";
const char * galaxy_zoo_toolset::header_filename = "header.txt";
const char * galaxy_zoo_toolset::testing_filename_pattern = "^(\\d+)\\.jpg$";

const float galaxy_zoo_toolset::max_rotation_angle_in_degrees = 180.0F;
const float galaxy_zoo_toolset::max_scale_factor = 1.1F; 
const float galaxy_zoo_toolset::max_shift = 2.0F;

const unsigned int galaxy_zoo_toolset::validating_rotation_sample_count = 24;

unsigned int galaxy_zoo_toolset::input_resized_width = 130;
unsigned int galaxy_zoo_toolset::input_resized_height = 130;
unsigned int galaxy_zoo_toolset::input_training_width = 100;
unsigned int galaxy_zoo_toolset::input_training_height = 100;
unsigned int galaxy_zoo_toolset::image_width = 68;
unsigned int galaxy_zoo_toolset::image_height = 68;

galaxy_zoo_toolset::galaxy_zoo_toolset(nnforge::factory_generator_smart_ptr factory)
	: nnforge::neural_network_toolset(factory)
{
}

galaxy_zoo_toolset::~galaxy_zoo_toolset()
{
}

std::map<unsigned int, float> galaxy_zoo_toolset::get_dropout_rate_map() const
{
	std::map<unsigned int, float> res;

	//res.insert(std::make_pair<unsigned int, float>(2 - (is_color_input ? 0 : 1), 0.1F));

	return res;
}

std::map<unsigned int, nnforge::weight_vector_bound> galaxy_zoo_toolset::get_weight_vector_bound_map() const
{
	std::map<unsigned int, nnforge::weight_vector_bound> res;

	float bound = 3.0F;
/*
	res.insert(std::make_pair(2 - (is_color_input ? 0 : 1), bound));
	res.insert(std::make_pair(5 - (is_color_input ? 0 : 1), bound));
	res.insert(std::make_pair(8 - (is_color_input ? 0 : 1), bound));
	res.insert(std::make_pair(11 - (is_color_input ? 0 : 1), bound));
	res.insert(std::make_pair(14 - (is_color_input ? 0 : 1), bound));
	res.insert(std::make_pair(17 - (is_color_input ? 0 : 1), bound));
	res.insert(std::make_pair(19 - (is_color_input ? 0 : 1), bound));
	res.insert(std::make_pair(21 - (is_color_input ? 0 : 1), bound));
*/
	return res;
}

nnforge::network_schema_smart_ptr galaxy_zoo_toolset::get_schema() const
{
	nnforge::network_schema_smart_ptr schema(new nnforge::network_schema());

	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::convolution_layer(std::vector<unsigned int>(2, 5), 3, 128))); // 64x64
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::maxout_layer(2)));
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::max_subsampling_layer(std::vector<unsigned int>(2,2)))); // 32x32
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::convolution_layer(std::vector<unsigned int>(2, 5), 64, 192))); // 28x28
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::maxout_layer(2)));
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::max_subsampling_layer(std::vector<unsigned int>(2,2)))); // 14x14
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::convolution_layer(std::vector<unsigned int>(2, 5), 96, 256))); // 10x10
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::maxout_layer(2)));
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::max_subsampling_layer(std::vector<unsigned int>(2,2)))); // 5x5
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::convolution_layer(std::vector<unsigned int>(2, 5), 128, 384))); // 1x1
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::maxout_layer(2)));
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::convolution_layer(std::vector<unsigned int>(2, 1), 192, 256))); // 1x1
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::maxout_layer(2)));
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::convolution_layer(std::vector<unsigned int>(2, 1), 128, 256))); // 1x1
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::maxout_layer(2)));
	schema->add_layer(nnforge::const_layer_smart_ptr(new nnforge::convolution_layer(std::vector<unsigned int>(2, 1), 128, output_neuron_count))); // 1x1

	return schema;
}

void galaxy_zoo_toolset::prepare_training_data()
{
	boost::filesystem::path input_training_folder_path = get_input_data_folder() / input_training_folder_name;
	std::cout << "Reading training images from " << input_training_folder_path.string() << std::endl;

	boost::filesystem::path input_training_file_path = get_input_data_folder() / training_filename;
	std::cout << "... and training values data from " << input_training_file_path.string() << std::endl;

	nnforge::layer_configuration_specific output_configuration;
	output_configuration.feature_map_count = output_neuron_count;
	output_configuration.dimension_sizes.push_back(1);
	output_configuration.dimension_sizes.push_back(1);

	nnforge::layer_configuration_specific input_configuration;
	input_configuration.feature_map_count = 3;
	input_configuration.dimension_sizes.push_back(input_training_width);
	input_configuration.dimension_sizes.push_back(input_training_height);

	boost::filesystem::ifstream file_input(input_training_file_path, std::ios_base::in);
	{
		std::string header;
		std::getline(file_input, header);
		std::vector<std::string> strs;
		boost::split(strs, header, boost::is_any_of(","));
		int val_count = strs.size() - 1;
		if (val_count != output_neuron_count)
			throw std::runtime_error((boost::format("Wrong number of questions encountered - %1%, expected - %2%") % val_count % output_neuron_count).str());

		boost::filesystem::path header_file_path = get_working_data_folder() / header_filename;
		boost::filesystem::ofstream testing_rec_labels_writer(header_file_path, std::ios_base::out | std::ios_base::trunc);
		testing_rec_labels_writer << header;
		std::cout << "Header dumped to " << header_file_path.string() << std::endl;
	}

	nnforge::supervised_data_stream_writer_smart_ptr training_data_writer;
	{
		boost::filesystem::path training_file_path = get_working_data_folder() / training_data_filename;
		std::cout << "Writing training data to " << training_file_path.string() << std::endl;
		nnforge_shared_ptr<std::ofstream> training_file(new boost::filesystem::ofstream(training_file_path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc));
		training_data_writer = nnforge::supervised_data_stream_writer_smart_ptr(new nnforge::supervised_data_stream_writer(
			training_file,
			input_configuration,
			output_configuration));
	}

	nnforge::supervised_data_stream_writer_smart_ptr validating_data_writer;
	if (is_training_with_validation())
	{
		boost::filesystem::path validating_file_path = get_working_data_folder() / validating_data_filename;
		std::cout << "... and writing validating data to " << validating_file_path.string() << std::endl;
		nnforge_shared_ptr<std::ofstream> validating_file(new boost::filesystem::ofstream(validating_file_path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc));
		validating_data_writer = nnforge::supervised_data_stream_writer_smart_ptr(new nnforge::supervised_data_stream_writer(
			validating_file,
			input_configuration,
			output_configuration));
	}

	std::vector<float> output_data(output_neuron_count);
	std::vector<unsigned char> input_data;
	unsigned int training_entry_count_written = 0;
	unsigned int validating_entry_count_written = 0;
	nnforge::random_generator gen = nnforge::rnd::get_random_generator();
	nnforge_uniform_real_distribution<float> dist(0.0F, 1.0F);
	while (true)
	{
		std::string str;
		std::getline(file_input, str);
		if (str.empty())
			break;
		std::vector<std::string> strs;
		boost::split(strs, str, boost::is_any_of(","));
		int val_count = strs.size() - 1;
		if (val_count != output_neuron_count)
			throw std::runtime_error((boost::format("Wrong number of questions encountered - %1%, expected - %2%") % val_count % output_neuron_count).str());

		boost::filesystem::path image_path = input_training_folder_path / (boost::format("%1%.jpg") % strs[0]).str();
		cv::Mat image_orig = cv::imread(image_path.string());
		cv::Mat image_resized = resize_image_for_training(image_orig);

		convert_to_input_format(image_resized, input_data);
		convert_to_output_format(strs.begin() + 1, output_data.begin());

		nnforge::supervised_data_stream_writer_smart_ptr current_data_writer;
		if (dist(gen) < reserved_for_validation)
		{
			current_data_writer = validating_data_writer;
			validating_entry_count_written++;
		}
		else
		{
			current_data_writer = training_data_writer;
			training_entry_count_written++;
		}

		current_data_writer->write(&(*input_data.begin()), &(*output_data.begin()));
	}

	std::cout << "Training entries written: " << training_entry_count_written << std::endl;
	if (is_training_with_validation())
		std::cout << "Validating entries written: " << validating_entry_count_written << std::endl;
}

void galaxy_zoo_toolset::prepare_testing_data()
{
	boost::filesystem::path input_testing_folder_path = get_input_data_folder() / input_testing_folder_name;
	std::cout << "Reading testing data from " << input_testing_folder_path.string() << std::endl;

	nnforge::layer_configuration_specific input_configuration;
	input_configuration.feature_map_count = 3;
	input_configuration.dimension_sizes.push_back(input_training_width);
	input_configuration.dimension_sizes.push_back(input_training_height);

	nnforge::unsupervised_data_stream_writer_smart_ptr testing_data_writer;
	{
		boost::filesystem::path testing_file_path = get_working_data_folder() / testing_unsupervised_data_filename;
		std::cout << "... and writing testing data to " << testing_file_path.string() << std::endl;
		nnforge_shared_ptr<std::ofstream> validating_file(new boost::filesystem::ofstream(testing_file_path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc));
		testing_data_writer = nnforge::unsupervised_data_stream_writer_smart_ptr(new nnforge::unsupervised_data_stream_writer(
			validating_file,
			input_configuration));
	}

	boost::filesystem::path testing_rec_ids_filepath = get_working_data_folder() / testing_rec_ids_filename;
	boost::filesystem::ofstream testing_rec_labels_writer(testing_rec_ids_filepath, std::ios_base::out | std::ios_base::trunc);
	std::cout << "... and testing labels to " << testing_rec_ids_filepath.string() << std::endl;

	nnforge_regex expression(testing_filename_pattern);
	nnforge_cmatch what;
	std::vector<unsigned char> input_data;
	unsigned int testing_entry_count_written = 0;
	for(boost::filesystem::directory_iterator it = boost::filesystem::directory_iterator(input_testing_folder_path); it != boost::filesystem::directory_iterator(); ++it)
	{
		boost::filesystem::path file_path = it->path();
		std::string file_name = file_path.filename().string();
		if (nnforge_regex_search(file_name.c_str(), what, expression))
		{
			std::string rec_id = std::string(what[1].first, what[1].second);

			cv::Mat image_orig = cv::imread(file_path.string());
			cv::Mat image_resized = resize_image_for_training(image_orig);

			testing_rec_labels_writer << rec_id << std::endl;

			convert_to_input_format(image_resized, input_data);
			testing_data_writer->write(&(*input_data.begin()));

			testing_entry_count_written++;
		}
	}

	std::cout << "Testing entries written: " << testing_entry_count_written << std::endl;
}

cv::Mat galaxy_zoo_toolset::resize_image_for_training(cv::Mat im) const
{
	cv::Mat dst_image(input_resized_height, input_resized_width, im.type());

	cv::resize(im, dst_image, dst_image.size(), 0.0, 0.0, CV_INTER_AREA);

	unsigned int start_y = (input_resized_height - input_training_height) / 2;
	unsigned int start_x = (input_resized_width - input_training_width) / 2;
	cv::Mat res_image = dst_image.rowRange(start_y, start_y + input_training_height).colRange(start_x, start_x + input_training_width);

	return res_image;
}

void galaxy_zoo_toolset::convert_to_input_format(
	cv::Mat3b image,
	std::vector<unsigned char>& input_data) const
{
	input_data.resize(image.rows * image.cols * 3);

	// Red
	std::transform(
		image.begin(),
		image.end(),
		input_data.begin(),
		vector_element_extractor<2U>());
	// Green
	std::transform(
		image.begin(),
		image.end(),
		input_data.begin() + (image.rows * image.cols),
		vector_element_extractor<1U>());
	// Blue
	std::transform(
		image.begin(),
		image.end(),
		input_data.begin() + (image.rows * image.cols * 2),
		vector_element_extractor<0U>());
}

void galaxy_zoo_toolset::convert_to_output_format(
	std::vector<std::string>::const_iterator input_it,
	std::vector<float>::iterator output_it) const
{
	for(std::vector<float>::iterator dst_it = output_it; dst_it != output_it + output_neuron_count; ++dst_it, ++input_it)
		*dst_it = static_cast<float>(atof(input_it->c_str()));
}

bool galaxy_zoo_toolset::is_training_with_validation() const
{
	return (reserved_for_validation > 0.0F);
}

std::vector<nnforge::data_transformer_smart_ptr> galaxy_zoo_toolset::get_input_data_transformer_list_for_training() const
{
	std::vector<nnforge::data_transformer_smart_ptr> res;

	res.push_back(nnforge::data_transformer_smart_ptr(new nnforge::distort_2d_data_transformer(
		max_rotation_angle_in_degrees,
		max_scale_factor,
		-max_shift,
		max_shift,
		-max_shift,
		max_shift,
		false,
		true)));

	std::vector<unsigned int> image_sizes;
	image_sizes.push_back(image_width);
	image_sizes.push_back(image_height);
	res.push_back(nnforge::data_transformer_smart_ptr(new nnforge::extract_data_transformer(
		image_sizes,
		image_sizes)));

	return res;
}

std::vector<nnforge::data_transformer_smart_ptr> galaxy_zoo_toolset::get_input_data_transformer_list_for_validating() const
{
	std::vector<nnforge::data_transformer_smart_ptr> res;

	/*
	std::vector<float> rotation_angle_in_degrees_list;
	for(unsigned int angle_sample = 0; angle_sample < validating_rotation_sample_count; ++angle_sample)
		rotation_angle_in_degrees_list.push_back(angle_sample * 360.0F / validating_rotation_sample_count);
	res.push_back(nnforge::data_transformer_smart_ptr(new nnforge::distort_2d_data_sampler_transformer(
		rotation_angle_in_degrees_list,
		std::vector<float>(1, 1.0F),
		std::vector<float>(1, 0.0F),
		std::vector<float>(1, 0.0F))));
	res.push_back(nnforge::data_transformer_smart_ptr(new nnforge::flip_2d_data_sampler_transformer(1)));
	*/

	std::vector<unsigned int> image_sizes;
	image_sizes.push_back(image_width);
	image_sizes.push_back(image_height);
	res.push_back(nnforge::data_transformer_smart_ptr(new nnforge::extract_data_transformer(
		image_sizes,
		image_sizes)));

	return res;
}

std::vector<nnforge::data_transformer_smart_ptr> galaxy_zoo_toolset::get_input_data_transformer_list_for_testing() const
{
	std::vector<nnforge::data_transformer_smart_ptr> res;

	std::vector<float> rotation_angle_in_degrees_list;
	for(unsigned int angle_sample = 0; angle_sample < validating_rotation_sample_count; ++angle_sample)
		rotation_angle_in_degrees_list.push_back(angle_sample * 360.0F / validating_rotation_sample_count);
	res.push_back(nnforge::data_transformer_smart_ptr(new nnforge::distort_2d_data_sampler_transformer(
		rotation_angle_in_degrees_list,
		std::vector<float>(1, 1.0F),
		std::vector<float>(1, 0.0F),
		std::vector<float>(1, 0.0F))));
	res.push_back(nnforge::data_transformer_smart_ptr(new nnforge::flip_2d_data_sampler_transformer(1)));

	std::vector<unsigned int> image_sizes;
	image_sizes.push_back(image_width);
	image_sizes.push_back(image_height);
	res.push_back(nnforge::data_transformer_smart_ptr(new nnforge::extract_data_transformer(
		image_sizes,
		image_sizes)));

	return res;
}

void galaxy_zoo_toolset::run_test_with_unsupervised_data(std::vector<nnforge::output_neuron_value_set_smart_ptr>& predicted_neuron_value_set_list)
{
	nnforge::output_neuron_value_set aggr_neuron_value_set(predicted_neuron_value_set_list, nnforge::output_neuron_value_set::merge_average);

	boost::filesystem::path testing_file_path = get_working_data_folder() / "output.csv";
	std::cout << "Writing testing results to " << testing_file_path.string() << std::endl;
	boost::filesystem::ofstream file_output(testing_file_path, std::ios_base::out | std::ios_base::trunc);
	file_output.exceptions(std::ostream::eofbit | std::ostream::failbit | std::ostream::badbit);

	boost::filesystem::path testing_rec_ids_filepath = get_working_data_folder() / testing_rec_ids_filename;
	boost::filesystem::ifstream file_input(testing_rec_ids_filepath, std::ios_base::in);

	// Dump the header
	{
		boost::filesystem::path header_filepath = get_working_data_folder() / header_filename;
		boost::filesystem::ifstream file_header_input(header_filepath, std::ios_base::in);
		std::string header;
		file_header_input >> header;
		file_output << header << std::endl;
	}

	nnforge::normalize_data_transformer_smart_ptr output_transformer = get_reverse_output_data_normalize_transformer();

	std::vector<std::vector<float> >::const_iterator it = aggr_neuron_value_set.neuron_value_list.begin();
	std::string str;
	while(true)
	{
		std::getline(file_input, str);
		boost::trim(str);

		if (str.empty())
			break;

		int rec_id = atol(str.c_str());
		file_output << rec_id;

		const std::vector<float>& value_list = *it;

		std::vector<std::pair<float, float> >::const_iterator mul_add_it = output_transformer->mul_add_list.begin();
		for(std::vector<float>::const_iterator src_it = value_list.begin(); src_it != value_list.end(); ++src_it, ++mul_add_it)
		{
			float src_val = *src_it;
			float transformed_val = src_val * mul_add_it->first + mul_add_it->second;
			float clipped_val = std::min(std::max(transformed_val, 0.0F), 1.0F);
			file_output << "," << (boost::format("%|1$.6f|") % clipped_val).str();
		}
		file_output << std::endl;

		++it;
	}
}

nnforge::network_output_type::output_type galaxy_zoo_toolset::get_network_output_type() const
{
	return nnforge::network_output_type::type_regression;
}

nnforge::testing_complete_result_set_visualizer_smart_ptr galaxy_zoo_toolset::get_validating_visualizer() const
{
	return nnforge::testing_complete_result_set_visualizer_smart_ptr(new galaxy_zoo_testing_complete_result_set_visualizer(get_reverse_output_data_normalize_transformer()));
}

std::vector<nnforge::data_transformer_smart_ptr> galaxy_zoo_toolset::get_output_data_transformer_list_for_training() const
{
	std::vector<nnforge::data_transformer_smart_ptr> res;

	res.push_back(get_output_data_normalize_transformer());

	return res;
}

std::vector<nnforge::data_transformer_smart_ptr> galaxy_zoo_toolset::get_output_data_transformer_list_for_validating() const
{
	std::vector<nnforge::data_transformer_smart_ptr> res;

	res.push_back(get_output_data_normalize_transformer());

	return res;
}

std::vector<nnforge::data_transformer_smart_ptr> galaxy_zoo_toolset::get_output_data_transformer_list_for_testing() const
{
	std::vector<nnforge::data_transformer_smart_ptr> res;

	res.push_back(get_output_data_normalize_transformer());

	return res;
}
