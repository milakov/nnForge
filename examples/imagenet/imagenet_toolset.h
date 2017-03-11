/*
 *  Copyright 2011-2016 Maxim Milakov
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

#include <nnforge/toolset.h>

#include <map>
#include <boost/filesystem/fstream.hpp>
#include <nnforge/nnforge.h>

class imagenet_toolset : public nnforge::toolset
{
public:
	imagenet_toolset(nnforge::factory_generator::ptr factory);

	virtual ~imagenet_toolset();

protected:
	virtual void prepare_training_data();

	virtual bool is_training_with_validation() const;

	virtual std::string get_class_name_by_id(unsigned int class_id) const;

	virtual nnforge::structured_data_reader::ptr get_structured_reader(
		const std::string& dataset_name,
		const std::string& layer_name,
		dataset_usage usage,
		std::shared_ptr<std::istream> in) const;

	virtual std::vector<nnforge::bool_option> get_bool_options();

	virtual std::vector<nnforge::int_option> get_int_options();

	virtual std::vector<nnforge::float_option> get_float_options();

	virtual std::vector<nnforge::string_option> get_string_options();

	virtual std::vector<nnforge::data_transformer::ptr> get_data_transformer_list(
		const std::string& dataset_name,
		const std::string& layer_name,
		dataset_usage usage) const;

	virtual void do_custom_action();

private:
	void prepare_true_randomized_training_data();

	void prepare_validating_data();

	unsigned int get_wnid_by_ilsvrc2014id(const std::string& ilsvrc2014id);

	unsigned int get_classid_by_wnid(unsigned int wnid) const;

	unsigned int get_wnid_by_classid(unsigned int classid) const;

	void load_cls_class_info();

	void write_supervised_data(
		const boost::filesystem::path& image_file_path,
		nnforge::varying_data_stream_writer& image_writer,
		unsigned int class_id,
		nnforge::structured_data_writer& label_writer);

	void create_resnet_dense_bottleneck_schema() const;

	void create_resnet_sparse_basic_schema() const;

	void create_resnet_dense_basic_schema() const;

	void add_resnet_bottleneck_block(
		std::vector<nnforge::layer::const_ptr>& layer_list,
		unsigned int& last_layer_feature_map_count,
		std::string& last_layer_name,
		unsigned int& bottleneck_major_block_id,
		char& bottleneck_minor_block_id,
		unsigned int& restored_feature_map_count,
		bool spatial_size_reduction) const;

	void add_resnet_sparse_bottleneck_block(
		std::vector<nnforge::layer::const_ptr>& layer_list,
		unsigned int& last_layer_feature_map_count,
		std::string& last_layer_name,
		unsigned int& bottleneck_major_block_id,
		char& bottleneck_minor_block_id,
		unsigned int& restored_feature_map_count,
		bool spatial_size_reduction,
		float beginning_sparsity_ratio,
		float sparsity_ratio) const;

	void add_resnet_sparse_basic_block(
		std::vector<nnforge::layer::const_ptr>& layer_list,
		unsigned int& last_layer_feature_map_count,
		std::string& last_layer_name,
		unsigned int& major_block_id,
		char& minor_block_id,
		unsigned int feature_map_count,
		bool spatial_size_reduction,
		float beginning_sparsity_ratio,
		float sparsity_ratio) const;

	void add_resnet_dense_basic_block(
		std::vector<nnforge::layer::const_ptr>& layer_list,
		unsigned int& last_layer_feature_map_count,
		std::string& last_layer_name,
		unsigned int& major_block_id,
		char& minor_block_id,
		unsigned int feature_map_count,
		bool spatial_size_reduction) const;

private:
	std::map<unsigned int, std::string> wnid_to_ilsvrc2014id_map;
	std::map<std::string, unsigned int> ilsvrc2014id_to_wnid_map;
	int epoch_count_in_image_net_training_set;
	std::string schema_create_type;

	static const char * cls_class_info_filename;
	static const char * training_images_folder_name;
	static const char * devkit_folder_name;
	static const char * devkit_data_folder_name;
	static const char * validation_ground_truth_file_name;
	static const char * validating_images_folder_name;
	static const char * ilsvrc2014id_pattern;
	static const char * training_image_filename_pattern;

	static const unsigned int class_count;
	static const unsigned int training_target_image_width;
	static const unsigned int training_target_image_height;
	static const unsigned int validating_image_size;

	bool rich_inference;
	int samples_x;
	int samples_y;
	float min_relative_target_area;
	float max_relative_target_area;
	float max_aspect_ratio_change;
	float max_brightness_shift;
	float max_contrast_shift;
	float max_saturation_shift;
	float max_lighting_shift;
	float min_elastic_deformation_intensity;
	float max_elastic_deformation_intensity;
	float min_elastic_deformation_smoothness;
	float max_elastic_deformation_smoothness;
	int sparse_feature_map_ratio;
	float connection_shuffle_ratio;

	static const unsigned int resnet_blocks[4];
};
