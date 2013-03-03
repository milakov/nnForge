/*
 *  Copyright 2011-2013 Maxim Milakov
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

#include "factory_generator.h"
#include "config_options.h"

#include "network_data_pusher.h"
#include "testing_complete_result_set_visualizer.h"

#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>

namespace nnforge
{
	class neural_network_toolset
	{
	public:
		neural_network_toolset(factory_generator_smart_ptr factory);

		virtual ~neural_network_toolset();

		// Returns true if action is specified
		bool parse(int argc, char* argv[]);

		virtual void do_action();

	protected:
		virtual std::vector<string_option> get_string_options();

		virtual std::vector<bool_option> get_bool_options();

		virtual std::vector<float_option> get_float_options();

		virtual std::vector<int_option> get_int_options();

		virtual void prepare_data() = 0;

		virtual network_schema_smart_ptr get_schema() = 0;

		virtual std::map<unsigned int, float> get_dropout_rate_map();

		virtual boost::filesystem::path get_input_data_folder() const;

		virtual boost::filesystem::path get_working_data_folder() const;

		virtual std::string get_class_name_by_id(unsigned int class_id) const;

		virtual void randomize_data();

		virtual void create();

		virtual network_tester_smart_ptr get_tester();

		virtual void validate(
			bool is_validate,
			bool infinite);

		virtual void validate_batch(bool is_validate);

		virtual void snapshot();

		virtual void snapshot_invalid();

		virtual void save_snapshot(
			const std::string& name,
			const std::vector<layer_configuration_specific_snapshot_smart_ptr>& data,
			bool folder_for_invalid = false);

		virtual std::vector<network_data_pusher_smart_ptr> get_validators_for_training(network_schema_smart_ptr schema);

		unsigned int get_starting_index_for_batch_training();

		virtual void train(bool batch = false);

		virtual void profile_updater();

		virtual void profile_hessian();

		virtual bool is_classifier() const;

		virtual bool is_normalize_input() const;

		virtual bool is_training_with_validation() const;

		virtual testing_complete_result_set_visualizer_smart_ptr get_testing_visualizer() const;

		static cv::Mat rotate_scale_shift(
			cv::Mat image,
			cv::Point2f rotation_center,
			float angle_in_degrees,
			float scale,
			float shift_x,
			float shift_y);

		// contrast: relative multiplication, about 1.0
		// brightness: change in luminocity for the middle lightness
		static cv::Mat change_brightness_and_contrast(
			cv::Mat image,
			float contrast,
			float brightness);

		static const char * training_data_filename;
		static const char * training_randomized_data_filename;
		static const char * validating_data_filename;
		static const char * testing_data_filename;
		static const char * testing_unsupervised_data_filename;
		static const char * scaling_params_filename;
		static const char * schema_filename;
		static const char * data_filename;
		static const char * data_trained_filename;
		static const char * snapshot_subfolder_name;
		static const char * snapshot_invalid_subfolder_name;
		static const char * batch_subfolder_name;
		static const char * trained_ann_index_extractor_pattern;

		network_tester_factory_smart_ptr tester_factory;
		network_updater_factory_smart_ptr updater_factory;
		hessian_calculator_factory_smart_ptr hessian_factory;

		std::string action;
		std::string snapshot_extension;
		unsigned int ann_count;
		unsigned int training_iteration_count;
		unsigned int snapshot_count;
		float training_speed;
		float training_speed_degradaton;
		float max_mu;
		float mu_increase_factor;
		unsigned int batch_offset;
		std::string snapshot_mode;
		unsigned int snapshot_video_fps;

		struct std_dev_helper
		{
			std_dev_helper(float average);

			float operator()(float x) const;

		private:
			float average;
		};

		static const float max_val_after_normalization;

	private:
		factory_generator_smart_ptr factory;

		boost::filesystem::path input_data_folder;
		boost::filesystem::path working_data_folder;

	private:
		neural_network_toolset();
		neural_network_toolset(const neural_network_toolset&);
	};
}
