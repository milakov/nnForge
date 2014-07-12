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

#pragma once

#include <nnforge/nnforge.h>
#include <opencv2/core/core.hpp>
#include <vector>
#include <string>

#include <nnforge/neural_network_toolset.h>

template<unsigned int index_id> class vector_element_extractor
{
public:
	vector_element_extractor()
	{
	}

	inline unsigned char operator()(cv::Vec3b x) const
	{
		return x[index_id];
	}
};

class galaxy_zoo_toolset : public nnforge::neural_network_toolset
{
public:
	galaxy_zoo_toolset(nnforge::factory_generator_smart_ptr factory);

	virtual ~galaxy_zoo_toolset();

protected:
	virtual std::map<unsigned int, float> get_dropout_rate_map() const;

	virtual nnforge::network_schema_smart_ptr get_schema() const;

	virtual void prepare_training_data();

	virtual void prepare_testing_data();

	virtual bool is_training_with_validation() const;

	virtual std::vector<nnforge::data_transformer_smart_ptr> get_input_data_transformer_list_for_training() const;

	virtual std::vector<nnforge::data_transformer_smart_ptr> get_input_data_transformer_list_for_validating() const;

	virtual std::vector<nnforge::data_transformer_smart_ptr> get_input_data_transformer_list_for_testing() const;

	virtual void run_test_with_unsupervised_data(std::vector<nnforge::output_neuron_value_set_smart_ptr>& predicted_neuron_value_set_list);

	virtual nnforge::network_output_type::output_type get_network_output_type() const;

	virtual nnforge::testing_complete_result_set_visualizer_smart_ptr get_validating_visualizer() const;

	virtual std::vector<nnforge::data_transformer_smart_ptr> get_output_data_transformer_list_for_training() const;

	virtual std::vector<nnforge::data_transformer_smart_ptr> get_output_data_transformer_list_for_validating() const;

	virtual std::vector<nnforge::data_transformer_smart_ptr> get_output_data_transformer_list_for_testing() const;

private:
	static const unsigned int output_neuron_count;
	static const float reserved_for_validation;

	static const char * input_training_folder_name;
	static const char * input_testing_folder_name;
	static const char * training_filename;
	static const char * testing_rec_ids_filename;
	static const char * header_filename;
	static const char * testing_filename_pattern;

	static unsigned int input_resized_width;
	static unsigned int input_resized_height;
	static unsigned int input_training_width;
	static unsigned int input_training_height;
	static unsigned int image_width;
	static unsigned int image_height;

	static const float max_rotation_angle_in_degrees;
	static const float max_scale_factor;
	static const float max_shift;
	static const float max_stretch_factor;

	static const unsigned int validating_rotation_sample_count;

	cv::Mat resize_image_for_training(cv::Mat im) const;

	void convert_to_input_format(
		cv::Mat3b image,
		std::vector<unsigned char>& input_data) const;

	void convert_to_output_format(
		std::vector<std::string>::const_iterator input_it,
		std::vector<float>::iterator output_it) const;
};
