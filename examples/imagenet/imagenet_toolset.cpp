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

#include "imagenet_toolset.h"

#include <algorithm>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <regex>

#include <nnforge/rnd.h>

#include "training_imagenet_raw_to_structured_data_transformer.h"
#include "validating_imagenet_raw_to_structured_data_transformer.h"

const char * imagenet_toolset::cls_class_info_filename = "cls_class_info.txt";
const char * imagenet_toolset::training_images_folder_name = "ILSVRC2012_img_train";
const char * imagenet_toolset::devkit_folder_name = "ILSVRC2014_devkit";
const char * imagenet_toolset::devkit_data_folder_name = "data";
const char * imagenet_toolset::validation_ground_truth_file_name = "ILSVRC2014_clsloc_validation_ground_truth.txt";
const char * imagenet_toolset::validating_images_folder_name = "ILSVRC2012_img_val";
const char * imagenet_toolset::ilsvrc2014id_pattern = "^n\\d{8}$";
const char * imagenet_toolset::training_image_filename_pattern = "^(n\\d{8})_(\\d+)\\.JPEG$";

const float imagenet_toolset::max_contrast_factor = 1.1F;
const float imagenet_toolset::max_brightness_shift = 0.0F;//0.05F;
const float imagenet_toolset::max_color_shift = 0.15F;

const unsigned int imagenet_toolset::class_count = 1000;
const unsigned int imagenet_toolset::training_target_image_width = 224;
const unsigned int imagenet_toolset::training_target_image_height = 224;
const unsigned int imagenet_toolset::validating_image_size = 256; // 256

const unsigned int imagenet_toolset::resnet10_blocks[4] = {3, 4, 6, 3};

imagenet_toolset::imagenet_toolset(nnforge::factory_generator::ptr factory)
	: nnforge::toolset(factory)
{
}

imagenet_toolset::~imagenet_toolset()
{
}

void imagenet_toolset::prepare_training_data()
{
	prepare_true_randomized_training_data();

	prepare_validating_data();
}

void imagenet_toolset::prepare_true_randomized_training_data()
{
	boost::filesystem::path training_images_folder_path = get_input_data_folder() / training_images_folder_name;
	std::cout << "Enumerating training images from " + training_images_folder_path.string() << "..." << std::endl;
	std::vector<std::pair<std::string, unsigned int> > ilsvrc2014id_localid_pair_list;
	{
		nnforge_regex folder_expression(ilsvrc2014id_pattern);
		nnforge_regex file_expression(training_image_filename_pattern);
		nnforge_cmatch what;
		for(boost::filesystem::directory_iterator it = boost::filesystem::directory_iterator(training_images_folder_path); it != boost::filesystem::directory_iterator(); ++it)
		{
			if (it->status().type() == boost::filesystem::directory_file)
			{
				boost::filesystem::path folder_path = it->path();
				std::string folder_name = folder_path.filename().string();
				if (nnforge_regex_match(folder_name, folder_expression))
				{
					const std::string& ilsvrc2014id = folder_name;
					unsigned int class_id = get_classid_by_wnid(get_wnid_by_ilsvrc2014id(ilsvrc2014id));
					for(boost::filesystem::directory_iterator it2 = boost::filesystem::directory_iterator(folder_path); it2 != boost::filesystem::directory_iterator(); ++it2)
					{
						if (it2->status().type() == boost::filesystem::regular_file)
						{
							boost::filesystem::path file_path = it2->path();
							std::string file_name = file_path.filename().string();
							if (nnforge_regex_search(file_name.c_str(), what, file_expression))
							{
								int localid = atol(std::string(what[2].first, what[2].second).c_str());
								ilsvrc2014id_localid_pair_list.push_back(std::make_pair(ilsvrc2014id, localid));
							}
						}
					}
				}
			}
		}
	}
	unsigned int total_training_image_count = static_cast<unsigned int>(ilsvrc2014id_localid_pair_list.size());
	std::cout << "Training images found: " << total_training_image_count << std::endl;

	nnforge::random_generator gen = nnforge::rnd::get_random_generator();

	nnforge::varying_data_stream_writer::ptr training_images_data_writer;
	{
		boost::filesystem::path training_images_file_path = get_working_data_folder() / "training_images.dt";
		std::cout << "Writing randomized training data (images) to " << training_images_file_path.string() << "..." << std::endl;
		nnforge_shared_ptr<std::ofstream> training_images_file_stream(new boost::filesystem::ofstream(training_images_file_path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc));
		training_images_data_writer = nnforge::varying_data_stream_writer::ptr(new nnforge::varying_data_stream_writer(training_images_file_stream));
	}

	nnforge::structured_data_writer::ptr training_labels_data_writer;
	{
		boost::filesystem::path training_labels_file_path = get_working_data_folder() / "training_labels.dt";
		std::cout << "Writing randomized training data (labels) to " << training_labels_file_path.string() << "..." << std::endl;
		nnforge_shared_ptr<std::ofstream> training_labels_file_stream(new boost::filesystem::ofstream(training_labels_file_path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc));
		nnforge::layer_configuration_specific config(class_count, std::vector<unsigned int>(2, 1));
		training_labels_data_writer = nnforge::structured_data_writer::ptr(new nnforge::structured_data_stream_writer(training_labels_file_stream, config));
	}

	for(unsigned int entry_written_count = 0; entry_written_count < total_training_image_count; ++entry_written_count)
	{
		nnforge_uniform_int_distribution<unsigned int> dist(0, static_cast<unsigned int>(ilsvrc2014id_localid_pair_list.size()) - 1);
		unsigned int index = dist(gen);

		std::pair<std::string, unsigned int> ilsvrc2014id_localid_pair = ilsvrc2014id_localid_pair_list[index];
		ilsvrc2014id_localid_pair_list[index] = ilsvrc2014id_localid_pair_list[ilsvrc2014id_localid_pair_list.size() - 1];
		ilsvrc2014id_localid_pair_list.pop_back();

		std::string filename = (boost::format("%1%_%2%.JPEG") % ilsvrc2014id_localid_pair.first % ilsvrc2014id_localid_pair.second).str();
		boost::filesystem::path image_file_path = training_images_folder_path / ilsvrc2014id_localid_pair.first / filename;
		int class_id = get_classid_by_wnid(get_wnid_by_ilsvrc2014id(ilsvrc2014id_localid_pair.first));

		write_supervised_data(
			image_file_path,
			*training_images_data_writer,
			class_id,
			*training_labels_data_writer);

		if (((entry_written_count + 1) % 50000) == 0)
			std::cout << (entry_written_count + 1) << " entries written" << std::endl;
	}
	std::cout << total_training_image_count << " entries written" << std::endl;
}

void imagenet_toolset::prepare_validating_data()
{
	std::vector<unsigned int> classid_list;
	{
		boost::filesystem::path validating_class_labels_filepath = get_input_data_folder() / devkit_folder_name / devkit_data_folder_name / validation_ground_truth_file_name;
		std::cout << "Reading ground truth labels from " + validating_class_labels_filepath.string() << "..." << std::endl;

		boost::filesystem::ifstream file_input(validating_class_labels_filepath, std::ios_base::in);

		std::string str;
		while (true)
		{
			std::getline(file_input, str);
			if (str.empty())
				break;

			unsigned int wnid = atol(str.c_str());
			unsigned int classid = get_classid_by_wnid(wnid);
			classid_list.push_back(classid);
		}
	}
	std::cout << classid_list.size() << " labels read\n";

	nnforge::varying_data_stream_writer::ptr validating_images_data_writer;
	{
		boost::filesystem::path validating_images_file_path = get_working_data_folder() / "validating_images.dt";
		std::cout << "Writing validating data (images) to " << validating_images_file_path.string() << "..." << std::endl;
		nnforge_shared_ptr<std::ofstream> validating_images_file_stream(new boost::filesystem::ofstream(validating_images_file_path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc));
		validating_images_data_writer = nnforge::varying_data_stream_writer::ptr(new nnforge::varying_data_stream_writer(validating_images_file_stream));
	}

	nnforge::structured_data_writer::ptr validating_labels_data_writer;
	{
		boost::filesystem::path validating_labels_file_path = get_working_data_folder() / "validating_labels.dt";
		std::cout << "Writing validating data (labels) to " << validating_labels_file_path.string() << "..." << std::endl;
		nnforge_shared_ptr<std::ofstream> validating_labels_file_stream(new boost::filesystem::ofstream(validating_labels_file_path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc));
		nnforge::layer_configuration_specific config(class_count, std::vector<unsigned int>(2, 1));
		validating_labels_data_writer = nnforge::structured_data_writer::ptr(new nnforge::structured_data_stream_writer(validating_labels_file_stream, config));
	}

	boost::filesystem::path validating_images_folder_path = get_input_data_folder() / validating_images_folder_name;
	for(int i = 0; i < classid_list.size(); ++i)
	{
		unsigned int class_id = classid_list[i];
		unsigned int image_id = i + 1;
		boost::filesystem::path image_file_path = validating_images_folder_path / (boost::format("ILSVRC2012_val_%|1$08d|.JPEG") % image_id).str();

		write_supervised_data(
			image_file_path,
			*validating_images_data_writer,
			class_id,
			*validating_labels_data_writer);
	}
	std::cout << classid_list.size() << " entries written" << std::endl;
}

void imagenet_toolset::write_supervised_data(
	const boost::filesystem::path& image_file_path,
	nnforge::varying_data_stream_writer& image_writer,
	unsigned int class_id,
	nnforge::structured_data_writer& label_writer)
{
	uintmax_t file_size = boost::filesystem::file_size(image_file_path);
	std::vector<unsigned char> image_content(file_size);
	{
		boost::filesystem::ifstream in(image_file_path, std::ios::binary);
		if (!in.read(reinterpret_cast<char *>(&(*image_content.begin())), file_size))
			throw std::runtime_error((boost::format("Error reading file %1%") % image_file_path.string()).str());
	}
	image_writer.raw_write(&(*image_content.begin()), image_content.size());

	std::vector<float> labels(class_count, 0.0F);
	labels[class_id] = 1.0F;
	label_writer.write(&labels[0]);
}

bool imagenet_toolset::is_training_with_validation() const
{
	return true;
}

unsigned int imagenet_toolset::get_classid_by_wnid(unsigned int wnid) const
{
	return wnid - 1;
}

unsigned int imagenet_toolset::get_wnid_by_classid(unsigned int classid) const
{
	return classid + 1;
}

std::string imagenet_toolset::get_class_name_by_id(unsigned int classid) const
{
	return (boost::format("%1%") % get_wnid_by_classid(classid)).str();
}

unsigned int imagenet_toolset::get_wnid_by_ilsvrc2014id(const std::string& ilsvrc2014id)
{
	if (wnid_to_ilsvrc2014id_map.empty())
		load_cls_class_info();

	std::map<std::string, unsigned int>::const_iterator it = ilsvrc2014id_to_wnid_map.find(ilsvrc2014id);
	if (it == ilsvrc2014id_to_wnid_map.end())
		throw std::runtime_error((boost::format("ilsvrc2014id '%1%' not found") % ilsvrc2014id).str());

	return it->second;
}

void imagenet_toolset::load_cls_class_info()
{
	wnid_to_ilsvrc2014id_map.clear();
	ilsvrc2014id_to_wnid_map.clear();

	boost::filesystem::path cls_class_info_filepath = get_working_data_folder() / cls_class_info_filename;
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

		if (strs.size() != 4)
			throw std::runtime_error((boost::format("Wrong number of fields in line %1%: %2%") % line_number % str).str());

		int wnid = atol(strs[0].c_str());

		wnid_to_ilsvrc2014id_map.insert(std::make_pair(wnid, strs[1]));
		ilsvrc2014id_to_wnid_map.insert(std::make_pair(strs[1], wnid));
	}

	if (wnid_to_ilsvrc2014id_map.empty())
		throw std::runtime_error((boost::format("No class info loaded from %1%") % cls_class_info_filepath.string()).str());
}

std::vector<nnforge::bool_option> imagenet_toolset::get_bool_options()
{
	std::vector<nnforge::bool_option> res = toolset::get_bool_options();

	res.push_back(nnforge::bool_option("rich_inference", &rich_inference, false, "Run multiple samples for each entry"));

	return res;
}

std::vector<nnforge::int_option> imagenet_toolset::get_int_options()
{
	std::vector<nnforge::int_option> res = toolset::get_int_options();

	res.push_back(nnforge::int_option("samples_x", &samples_x, 4, "Run multiple samples (in x direction) for each entry"));
	res.push_back(nnforge::int_option("samples_y", &samples_y, 4, "Run multiple samples (in y direction) for each entry"));
	res.push_back(nnforge::int_option("training_min_image_size", &training_min_image_size, 224, "Minimum short side of the image from which training sample is extracted"));
	res.push_back(nnforge::int_option("training_max_image_size", &training_max_image_size, 288, "Maximum short side of the image from which training sample is extracted"));

	return res;
}

std::vector<nnforge::float_option> imagenet_toolset::get_float_options()
{
	std::vector<nnforge::float_option> res = toolset::get_float_options();

	res.push_back(nnforge::float_option("max_aspect_ratio_change", &max_aspect_ratio_change, 1.0F, "The maximum aspect ration change during training"));

	return res;
}

nnforge::structured_data_reader::ptr imagenet_toolset::get_structured_reader(
	const std::string& dataset_name,
	const std::string& layer_name,
	dataset_usage usage,
	nnforge_shared_ptr<std::istream> in) const
{
	if (layer_name == "images")
	{
		nnforge::raw_data_reader::ptr raw_reader(new nnforge::varying_data_stream_reader(in));
		nnforge::raw_to_structured_data_transformer::ptr transformer;
		if (dataset_name == "training")
		{
			if (usage == dataset_usage_check_gradient)
			{
				std::vector<std::pair<float, float> > position_list(1, std::make_pair(0.5F, 0.5F));
				transformer = nnforge::raw_to_structured_data_transformer::ptr(new validating_imagenet_raw_to_structured_data_transformer(
					validating_image_size,
					training_target_image_width,
					training_target_image_height,
					position_list));
			}
			else
			{
				transformer = nnforge::raw_to_structured_data_transformer::ptr(new training_imagenet_raw_to_structured_data_transformer(
					training_min_image_size,
					training_max_image_size,
					training_target_image_width,
					training_target_image_height,
					max_aspect_ratio_change));
			}
		}
		else if (dataset_name == "validating")
		{
			std::vector<std::pair<float, float> > position_list;
			if (rich_inference)
			{
				for(int sample_id_x = 0; sample_id_x < samples_x; ++sample_id_x)
				{
					float pos_x = 0.5F;
					if (samples_x > 1)
						pos_x = sample_id_x / static_cast<float>(samples_x - 1);

					for(int sample_id_y = 0; sample_id_y < samples_y; ++sample_id_y)
					{
						float pos_y = 0.5F;
						if (samples_y > 1)
							pos_y = sample_id_y / static_cast<float>(samples_y - 1);

						position_list.push_back(std::make_pair(pos_x, pos_y));
					}
				}
			}
			else
			{
				position_list.push_back(std::make_pair(0.5F, 0.5F));
			}

			transformer = nnforge::raw_to_structured_data_transformer::ptr(new validating_imagenet_raw_to_structured_data_transformer(
				validating_image_size,
				training_target_image_width,
				training_target_image_height,
				position_list));
		}
		return nnforge::structured_data_reader::ptr(new nnforge::structured_from_raw_data_reader(raw_reader, transformer));
	}
	else
		return toolset::get_structured_reader(dataset_name, layer_name, usage, in);
}

std::vector<nnforge::data_transformer::ptr> imagenet_toolset::get_data_transformer_list(
	const std::string& dataset_name,
	const std::string& layer_name,
	dataset_usage usage) const
{
	std::vector<nnforge::data_transformer::ptr> res;

	if (usage != dataset_usage_check_gradient)
	{
		if (layer_name == "images")
		{
			res.push_back(get_normalize_data_transformer(layer_name));
			if ((dataset_name == "training") && (usage != dataset_usage_create_normalizer))
			{
				res.push_back(nnforge::data_transformer::ptr(new nnforge::distort_2d_data_transformer(
					0.0F,
					1.0F,
					0.0F,
					0.0F,
					0.0F,
					0.0F,
					false,
					true,
					1.0F,
					std::numeric_limits<float>::max())));
				res.push_back(nnforge::data_transformer::ptr(new nnforge::uniform_intensity_data_transformer(
					std::vector<float>(3, -max_color_shift),
					std::vector<float>(3, max_color_shift))));
			}
			else if (dataset_name == "validating")
			{
				if (rich_inference)
				{
					res.push_back(nnforge::data_transformer::ptr(new nnforge::distort_2d_data_sampler_transformer(
						std::vector<float>(1, 0.0F),
						std::vector<float>(1, 1.0F),
						std::vector<float>(1, 0.0F),
						std::vector<float>(1, 0.0F),
						std::vector<std::pair<float, float> >(1, std::make_pair(1.0F, 0.0F)),
						std::vector<std::pair<float, float> >(1, std::make_pair(std::numeric_limits<float>::max(), 0.0F)),
						false,
						true)));
				}
			}
		}
	}

	return res;
}

void imagenet_toolset::do_custom_action()
{
	if (action == "create_resnet_schema")
	{
		create_resnet_schema();
	}
	else
		toolset::do_custom_action();
}

void imagenet_toolset::create_resnet_schema() const
{
	std::vector<nnforge::layer::const_ptr> layer_list;

	nnforge::layer::ptr images_data_layer(new nnforge::data_layer());
	images_data_layer->instance_name = "images";
	layer_list.push_back(images_data_layer);

	nnforge::layer::ptr labels_data_layer(new nnforge::data_layer());
	labels_data_layer->instance_name = "labels";
	layer_list.push_back(labels_data_layer);

	nnforge::layer::ptr conv1_layer(new nnforge::convolution_layer(std::vector<unsigned int>(2, 7), 3, 64, std::vector<unsigned int>(2, 3), std::vector<unsigned int>(2, 3), std::vector<unsigned int>(2, 2), false));
	conv1_layer->instance_name = "conv1";
	conv1_layer->input_layer_instance_names.push_back("images");
	layer_list.push_back(conv1_layer);

	nnforge::layer::ptr conv1_bn_layer(new nnforge::batch_norm_layer(64));
	conv1_bn_layer->instance_name = "conv1_bn";
	conv1_bn_layer->input_layer_instance_names.push_back("conv1");
	layer_list.push_back(conv1_bn_layer);

	nnforge::layer::ptr relu1_layer(new nnforge::rectified_linear_layer());
	relu1_layer->instance_name = "relu1";
	relu1_layer->input_layer_instance_names.push_back("conv1_bn");
	layer_list.push_back(relu1_layer);

	nnforge::layer::ptr pool1_layer(new nnforge::max_subsampling_layer(std::vector<unsigned int>(2, 2)));
	pool1_layer->instance_name = "pool1";
	pool1_layer->input_layer_instance_names.push_back("relu1");
	layer_list.push_back(pool1_layer);

	unsigned int last_layer_feature_map_count = 64;
	std::string last_layer_name = "pool1";
	unsigned int bottleneck_major_block_id = 2;
	char bottleneck_minor_block_id = 'a';
	unsigned int restored_feature_map_count = 256;

	for(unsigned int resnet_spatial_block_id = 0; resnet_spatial_block_id < sizeof(resnet10_blocks) / sizeof(resnet10_blocks[0]); ++resnet_spatial_block_id)
		for(unsigned int resnet_block_id = 0; resnet_block_id < resnet10_blocks[resnet_spatial_block_id]; ++resnet_block_id)
			add_resnet_bottleneck_block(layer_list, last_layer_feature_map_count, last_layer_name, bottleneck_major_block_id, bottleneck_minor_block_id, restored_feature_map_count, (resnet_block_id == 0) && (resnet_spatial_block_id != 0));

	std::string avg_pool_layer_name = (boost::format("pool%1%") % bottleneck_major_block_id).str(); 
	nnforge::layer::ptr avg_pool_layer(new nnforge::average_subsampling_layer(std::vector<unsigned int>(2, 7)));
	avg_pool_layer->instance_name = avg_pool_layer_name;
	avg_pool_layer->input_layer_instance_names.push_back(last_layer_name);
	layer_list.push_back(avg_pool_layer);

	nnforge::layer::ptr logits_layer(new nnforge::convolution_layer(std::vector<unsigned int>(2, 1), last_layer_feature_map_count, 1000, std::vector<unsigned int>(), std::vector<unsigned int>(), std::vector<unsigned int>(), false));
	logits_layer->instance_name = "logits";
	logits_layer->input_layer_instance_names.push_back(avg_pool_layer_name);
	layer_list.push_back(logits_layer);

	nnforge::layer::ptr prob_layer(new nnforge::softmax_layer());
	prob_layer->instance_name = "prob";
	prob_layer->input_layer_instance_names.push_back("logits");
	layer_list.push_back(prob_layer);

	nnforge::layer::ptr nll_layer(new nnforge::negative_log_likelihood_layer());
	nll_layer->instance_name = "NLL";
	nll_layer->input_layer_instance_names.push_back("prob");
	nll_layer->input_layer_instance_names.push_back("labels");
	layer_list.push_back(nll_layer);

	nnforge::layer::ptr accuracy_layer(new nnforge::accuracy_layer(5));
	accuracy_layer->instance_name = "Accuracy";
	accuracy_layer->input_layer_instance_names.push_back("prob");
	accuracy_layer->input_layer_instance_names.push_back("labels");
	layer_list.push_back(accuracy_layer);

	nnforge::network_schema schema(layer_list);
	schema.name = "ResNet-50";
	boost::filesystem::ofstream out(get_working_data_folder() / "schema_resnet50.txt");
	schema.write_proto(out);
}

void imagenet_toolset::add_resnet_bottleneck_block(
	std::vector<nnforge::layer::const_ptr>& layer_list,
	unsigned int& last_layer_feature_map_count,
	std::string& last_layer_name,
	unsigned int& bottleneck_major_block_id,
	char& bottleneck_minor_block_id,
	unsigned int& restored_feature_map_count,
	bool spatial_size_reduction) const
{
	if (spatial_size_reduction)
	{
		restored_feature_map_count *= 2;
		++bottleneck_major_block_id;
		bottleneck_minor_block_id = 'a';
	}

	unsigned int bottleneck_feature_map_count = restored_feature_map_count / 4;

	std::string block_name = (boost::format("res%1%%2%") % bottleneck_major_block_id % bottleneck_minor_block_id).str();

	std::string to_add_layer_name = last_layer_name;
	if (spatial_size_reduction || (last_layer_feature_map_count != restored_feature_map_count))
	{
		std::string shortcut_layer_name = block_name + "_shortcut";
		nnforge::layer::ptr shortcut_layer(new nnforge::convolution_layer(std::vector<unsigned int>(2, 1), last_layer_feature_map_count, restored_feature_map_count, std::vector<unsigned int>(2, 0), std::vector<unsigned int>(2, 0), std::vector<unsigned int>(2, spatial_size_reduction ? 2 : 1), false));
		shortcut_layer->instance_name = shortcut_layer_name;
		shortcut_layer->input_layer_instance_names.push_back(last_layer_name);
		layer_list.push_back(shortcut_layer);
		std::string shortcut_bn_layer_name = block_name + "_shortcut_bn";
		nnforge::layer::ptr shortcut_bn_layer(new nnforge::batch_norm_layer(restored_feature_map_count));
		shortcut_bn_layer->instance_name = shortcut_bn_layer_name;
		shortcut_bn_layer->input_layer_instance_names.push_back(shortcut_layer_name);
		layer_list.push_back(shortcut_bn_layer);
		to_add_layer_name = shortcut_bn_layer_name;
	}

	std::string reduce_fm_layer_name = block_name + "_reduce_fm";
	std::string reduce_fm_bn_layer_name = reduce_fm_layer_name + "_bn";
	std::string reduce_fm_relu_layer_name = reduce_fm_layer_name + "_relu";
	{
		nnforge::layer::ptr reduce_fm_layer(new nnforge::convolution_layer(std::vector<unsigned int>(2, 1), last_layer_feature_map_count, bottleneck_feature_map_count, std::vector<unsigned int>(2, 0), std::vector<unsigned int>(2, 0), std::vector<unsigned int>(2, spatial_size_reduction ? 2 : 1), false));
		reduce_fm_layer->instance_name = reduce_fm_layer_name;
		reduce_fm_layer->input_layer_instance_names.push_back(last_layer_name);
		layer_list.push_back(reduce_fm_layer);
		nnforge::layer::ptr reduce_fm_bn_layer(new nnforge::batch_norm_layer(bottleneck_feature_map_count));
		reduce_fm_bn_layer->instance_name = reduce_fm_bn_layer_name;
		reduce_fm_bn_layer->input_layer_instance_names.push_back(reduce_fm_layer_name);
		layer_list.push_back(reduce_fm_bn_layer);
		nnforge::layer::ptr reduce_fm_relu_layer(new nnforge::rectified_linear_layer());
		reduce_fm_relu_layer->instance_name = reduce_fm_relu_layer_name;
		reduce_fm_relu_layer->input_layer_instance_names.push_back(reduce_fm_bn_layer_name);
		layer_list.push_back(reduce_fm_relu_layer);
	}

	std::string bottleneck_layer_name = block_name + "_bottleneck";
	std::string bottleneck_bn_layer_name = bottleneck_layer_name + "_bn";
	std::string bottleneck_relu_layer_name = bottleneck_layer_name + "_relu";
	{
		nnforge::layer::ptr bottleneck_layer(new nnforge::convolution_layer(std::vector<unsigned int>(2, 3), bottleneck_feature_map_count, bottleneck_feature_map_count, std::vector<unsigned int>(2, 1), std::vector<unsigned int>(2, 1), std::vector<unsigned int>(2, 1), false));
		bottleneck_layer->instance_name = bottleneck_layer_name;
		bottleneck_layer->input_layer_instance_names.push_back(reduce_fm_relu_layer_name);
		layer_list.push_back(bottleneck_layer);
		nnforge::layer::ptr bottleneck_bn_layer(new nnforge::batch_norm_layer(bottleneck_feature_map_count));
		bottleneck_bn_layer->instance_name = bottleneck_bn_layer_name;
		bottleneck_bn_layer->input_layer_instance_names.push_back(bottleneck_layer_name);
		layer_list.push_back(bottleneck_bn_layer);
		nnforge::layer::ptr bottleneck_relu_layer(new nnforge::rectified_linear_layer());
		bottleneck_relu_layer->instance_name = bottleneck_relu_layer_name;
		bottleneck_relu_layer->input_layer_instance_names.push_back(bottleneck_bn_layer_name);
		layer_list.push_back(bottleneck_relu_layer);
	}

	std::string restore_fm_layer_name = block_name + "_restore_fm";
	std::string restore_fm_bn_layer_name = restore_fm_layer_name + "_bn";
	{
		nnforge::layer::ptr restore_fm_layer(new nnforge::convolution_layer(std::vector<unsigned int>(2, 1), bottleneck_feature_map_count, restored_feature_map_count, std::vector<unsigned int>(2, 0), std::vector<unsigned int>(2, 0), std::vector<unsigned int>(2, 1), false));
		restore_fm_layer->instance_name = restore_fm_layer_name;
		restore_fm_layer->input_layer_instance_names.push_back(bottleneck_relu_layer_name);
		layer_list.push_back(restore_fm_layer);
		nnforge::layer::ptr restore_fm_bn_layer(new nnforge::batch_norm_layer(restored_feature_map_count));
		restore_fm_bn_layer->instance_name = restore_fm_bn_layer_name;
		restore_fm_bn_layer->input_layer_instance_names.push_back(restore_fm_layer_name);
		layer_list.push_back(restore_fm_bn_layer);
	}

	std::string add_layer_name = block_name;
	std::string add_relu_layer_name = add_layer_name + "_relu";
	{
		nnforge::layer::ptr add_layer(new nnforge::add_layer());
		add_layer->instance_name = add_layer_name;
		add_layer->input_layer_instance_names.push_back(to_add_layer_name);
		add_layer->input_layer_instance_names.push_back(restore_fm_bn_layer_name);
		layer_list.push_back(add_layer);
		nnforge::layer::ptr add_relu_layer(new nnforge::rectified_linear_layer());
		add_relu_layer->instance_name = add_relu_layer_name;
		add_relu_layer->input_layer_instance_names.push_back(add_layer_name);
		layer_list.push_back(add_relu_layer);
	}

	last_layer_feature_map_count = restored_feature_map_count;
	last_layer_name = add_relu_layer_name;
	++bottleneck_minor_block_id;
}
