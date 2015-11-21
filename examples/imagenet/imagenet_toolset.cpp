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
const unsigned int imagenet_toolset::training_min_image_size = 224; //256
const unsigned int imagenet_toolset::training_max_image_size = 288; //256
const unsigned int imagenet_toolset::training_target_image_width = 224;
const unsigned int imagenet_toolset::training_target_image_height = 224;
const unsigned int imagenet_toolset::validating_image_size = 256; // 256

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
/*
std::vector<nnforge::data_transformer_smart_ptr> imagenet_toolset::get_input_data_transformer_list_for_training() const
{
	std::vector<nnforge::data_transformer_smart_ptr> res;
	
		
	res.push_back(nnforge::data_transformer_smart_ptr(new nnforge::convert_data_type_transformer()));
	res.push_back(get_input_data_normalize_transformer());


	return res;
}

std::vector<nnforge::data_transformer_smart_ptr> imagenet_toolset::get_input_data_transformer_list_for_validating() const
{
	std::vector<nnforge::data_transformer_smart_ptr> res;

	if (rich_inference)
		res.push_back(nnforge::data_transformer_smart_ptr(new nnforge::flip_2d_data_sampler_transformer(1)));

	res.push_back(nnforge::data_transformer_smart_ptr(new nnforge::convert_data_type_transformer()));
	res.push_back(get_input_data_normalize_transformer());

	return res;
}

std::vector<nnforge::data_transformer_smart_ptr> imagenet_toolset::get_input_data_transformer_list_for_testing() const
{
	std::vector<nnforge::data_transformer_smart_ptr> res;

	if (rich_inference)
		res.push_back(nnforge::data_transformer_smart_ptr(new nnforge::flip_2d_data_sampler_transformer(1)));

	res.push_back(nnforge::data_transformer_smart_ptr(new nnforge::convert_data_type_transformer()));
	res.push_back(get_input_data_normalize_transformer());

	return res;
}

void imagenet_toolset::run_test_with_unsupervised_data(const nnforge::output_neuron_value_set& neuron_value_set)
{

}
*/
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

/*
std::vector<nnforge::network_data_pusher_smart_ptr> imagenet_toolset::get_validators_for_training(nnforge::network_schema_smart_ptr schema)
{
	std::vector<nnforge::network_data_pusher_smart_ptr> res = neural_network_toolset::get_validators_for_training(schema);

	nnforge_shared_ptr<std::istream> validating_data_stream(new boost::filesystem::ifstream(get_working_data_folder() / validating_data_filename, std::ios_base::in | std::ios_base::binary));
	nnforge::supervised_data_reader_smart_ptr current_reader = get_validating_reader(validating_data_stream, true);
	{
		nnforge::supervised_data_reader_smart_ptr new_reader(new nnforge::supervised_transformed_input_data_reader(current_reader, nnforge::data_transformer_smart_ptr(new nnforge::flip_2d_data_sampler_transformer(1))));
		current_reader = new_reader;
	}
	{
		nnforge::supervised_data_reader_smart_ptr new_reader(new nnforge::supervised_transformed_input_data_reader(current_reader, nnforge::data_transformer_smart_ptr(new nnforge::convert_data_type_transformer())));
		current_reader = new_reader;
	}
	{
		nnforge::supervised_data_reader_smart_ptr new_reader(new nnforge::supervised_transformed_input_data_reader(current_reader, get_input_data_normalize_transformer()));
		current_reader = new_reader;
	}

	res.push_back(nnforge::network_data_pusher_smart_ptr(new nnforge::validate_progress_network_data_pusher(
		tester_factory->create(schema),
		current_reader,
		get_validating_visualizer(),
		get_error_function(),
		current_reader->get_sample_count(),
		enrich_validation_report_frequency)));

	return res;
}

nnforge::supervised_data_reader_smart_ptr imagenet_toolset::get_validating_reader(
	nnforge_shared_ptr<std::istream> validating_data_stream,
	bool enriched) const
{
	std::vector<std::pair<float, float> > position_list;
	if (enriched)
	{
		float start_y;
		float step_y;
		if (overlapping_samples_y > 1)
		{
			start_y = (1.0F - sample_coverage_y) * 0.5F;
			step_y = sample_coverage_y / static_cast<float>(overlapping_samples_y - 1);
		}
		else
		{
			start_y = 0.5F;
			step_y = 0.0F;
		}
		float start_x;
		float step_x;
		if (overlapping_samples_x > 1)
		{
			start_x = (1.0F - sample_coverage_x) * 0.5F;
			step_x = sample_coverage_x / static_cast<float>(overlapping_samples_x - 1);
		}
		else
		{
			start_x = 0.5F;
			step_x = 0.0F;
		}
		for(unsigned int pos_y = 0; pos_y < overlapping_samples_y; ++pos_y)
		{
			float pos_y_f = static_cast<float>(pos_y) * step_y + start_y;
			for(unsigned int pos_x = 0; pos_x < overlapping_samples_x; ++pos_x)
			{
				float pos_x_f = static_cast<float>(pos_x) * step_x + start_x;
				position_list.push_back(std::make_pair(pos_x_f, pos_y_f));
			}
		}
	}
	else
	{
		position_list.push_back(std::make_pair(0.5F, 0.5F));
	}

	nnforge::supervised_data_reader_smart_ptr res(new nnforge::supervised_image_data_sampler_stream_reader(
		validating_data_stream,
		training_image_original_width,
		training_image_original_height,
		training_image_width,
		training_image_height,
		class_count,
		false,
		true,
		128,
		position_list));

	return res;
}
*/

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

	return res;
}

nnforge::structured_data_reader::ptr imagenet_toolset::get_structured_reader(
	const std::string& dataset_name,
	const std::string& layer_name,
	nnforge_shared_ptr<std::istream> in) const
{
	if (layer_name == "images")
	{
		nnforge::raw_data_reader::ptr raw_reader(new nnforge::varying_data_stream_reader(in));
		nnforge::raw_to_structured_data_transformer::ptr transformer;
		if (dataset_name == "training")
		{
			transformer = nnforge::raw_to_structured_data_transformer::ptr(new training_imagenet_raw_to_structured_data_transformer(
				training_min_image_size,
				training_max_image_size,
				training_target_image_width,
				training_target_image_height));
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
		return toolset::get_structured_reader(dataset_name, layer_name, in);
}

std::vector<nnforge::data_transformer::ptr> imagenet_toolset::get_data_transformer_list(
	const std::string& dataset_name,
	const std::string& layer_name,
	dataset_usage usage) const
{
	std::vector<nnforge::data_transformer::ptr> res;

	if ((layer_name == "images") && (usage != dataset_usage_create_normalizer))
	{
		if (dataset_name == "training")
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
			res.push_back(nnforge::data_transformer::ptr(new nnforge::intensity_2d_data_transformer(
				max_contrast_factor,
				max_brightness_shift)));
			res.push_back(get_normalize_data_transformer(layer_name));
			res.push_back(nnforge::data_transformer::ptr(new nnforge::uniform_intensity_data_transformer(
				std::vector<float>(3, -max_color_shift),
				std::vector<float>(3, max_color_shift))));
		}
		else if (dataset_name == "validating")
		{
			res.push_back(get_normalize_data_transformer(layer_name));
		}
	}

	return res;
}
