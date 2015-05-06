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

#pragma once

#include <nnforge/neural_network_toolset.h>

#include <map>
#include <boost/filesystem/fstream.hpp>
#include <nnforge/nnforge.h>

class imagenet_toolset : public nnforge::neural_network_toolset
{
public:
	imagenet_toolset(nnforge::factory_generator_smart_ptr factory);

	virtual ~imagenet_toolset();

protected:
	virtual void prepare_training_data();

	virtual void prepare_testing_data();

	virtual bool is_training_with_validation() const;

	virtual std::vector<nnforge::data_transformer_smart_ptr> get_input_data_transformer_list_for_training() const;

	virtual std::vector<nnforge::data_transformer_smart_ptr> get_input_data_transformer_list_for_validating() const;

	virtual std::vector<nnforge::data_transformer_smart_ptr> get_input_data_transformer_list_for_testing() const;

	virtual void run_test_with_unsupervised_data(const nnforge::output_neuron_value_set& neuron_value_set);

	virtual std::string get_class_name_by_id(unsigned int class_id) const;

	virtual nnforge::supervised_data_reader_smart_ptr get_initial_data_reader_for_normalizing() const;

	virtual nnforge::supervised_data_reader_smart_ptr get_initial_data_reader_for_training(bool force_deterministic) const;

	virtual nnforge::supervised_data_reader_smart_ptr get_initial_data_reader_for_validating() const;

	virtual nnforge::const_error_function_smart_ptr get_error_function() const;

	virtual unsigned int get_classifier_visualizer_top_n() const;

	virtual std::vector<nnforge::network_data_pusher_smart_ptr> get_validators_for_training(nnforge::network_schema_smart_ptr schema);

	virtual nnforge::supervised_data_reader_smart_ptr get_validating_reader(
		nnforge_shared_ptr<std::istream> validating_data_stream,
		bool enriched) const;

private:
	void prepare_randomized_training_data();

	void prepare_true_randomized_training_data();

	void prepare_validating_data();

	unsigned int get_wnid_by_ilsvrc2014id(const std::string& ilsvrc2014id);

	unsigned int get_classid_by_wnid(unsigned int wnid) const;

	unsigned int get_wnid_by_classid(unsigned int classid) const;

	void load_cls_class_info();

	void write_supervised_data(
		const boost::filesystem::path& image_file_path,
		nnforge::varying_data_stream_writer& writer,
		unsigned int class_id);

	virtual std::vector<nnforge::bool_option> get_bool_options();

private:
	std::map<unsigned int, std::string> wnid_to_ilsvrc2014id_map;
	std::map<std::string, unsigned int> ilsvrc2014id_to_wnid_map;
	int epoch_count_in_image_net_training_set;

	static const char * cls_class_info_filename;
	static const char * training_images_folder_name;
	static const char * devkit_folder_name;
	static const char * devkit_data_folder_name;
	static const char * validation_ground_truth_file_name;
	static const char * validating_images_folder_name;
	static const char * ilsvrc2014id_pattern;
	static const char * training_image_filename_pattern;

	static const float max_contrast_factor;
	static const float max_brightness_shift;
	static const float max_color_shift;

	static const unsigned int class_count;
	static const unsigned int training_image_width;
	static const unsigned int training_image_height;
	static const unsigned int training_image_original_width;
	static const unsigned int training_image_original_height;

	static const unsigned int enrich_validation_report_frequency;
	static const unsigned int overlapping_samples_x;
	static const unsigned int overlapping_samples_y;
	static const float sample_coverage_x;
	static const float sample_coverage_y;

	bool rich_inference;
};
