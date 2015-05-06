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

#include <nnforge/supervised_transformed_input_data_reader.h>

#include <algorithm>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <regex>

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
const unsigned int imagenet_toolset::training_image_width = 224;
const unsigned int imagenet_toolset::training_image_height = 224;
const unsigned int imagenet_toolset::training_image_original_width = 256;
const unsigned int imagenet_toolset::training_image_original_height = 256;

const unsigned int imagenet_toolset::enrich_validation_report_frequency = 500;
const unsigned int imagenet_toolset::overlapping_samples_x = 4;
const unsigned int imagenet_toolset::overlapping_samples_y = 4;
const float imagenet_toolset::sample_coverage_x = 1.0F;
const float imagenet_toolset::sample_coverage_y = 1.0F;

imagenet_toolset::imagenet_toolset(nnforge::factory_generator_smart_ptr factory)
	: nnforge::neural_network_toolset(factory)
{
}

imagenet_toolset::~imagenet_toolset()
{
}

nnforge::const_error_function_smart_ptr imagenet_toolset::get_error_function() const
{
	return nnforge::const_error_function_smart_ptr(new nnforge::negative_log_likelihood_error_function());
}

void imagenet_toolset::prepare_training_data()
{
	prepare_true_randomized_training_data();

	prepare_validating_data();
}

void imagenet_toolset::prepare_randomized_training_data()
{
	boost::filesystem::path training_images_folder_path = get_input_data_folder() / training_images_folder_name;
	unsigned int total_training_image_count = 0;
	std::cout << "Enumerating training images from " + training_images_folder_path.string() << "..." << std::endl;
	std::map<std::string, std::vector<unsigned int> > ilsvrc2014id_to_localid_list_map;
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
					std::vector<unsigned int>& localid_list =  ilsvrc2014id_to_localid_list_map.insert(std::make_pair(ilsvrc2014id, std::vector<unsigned int>())).first->second;
					for(boost::filesystem::directory_iterator it2 = boost::filesystem::directory_iterator(folder_path); it2 != boost::filesystem::directory_iterator(); ++it2)
					{
						if (it2->status().type() == boost::filesystem::regular_file)
						{
							boost::filesystem::path file_path = it2->path();
							std::string file_name = file_path.filename().string();
							if (nnforge_regex_search(file_name.c_str(), what, file_expression))
							{
								std::string ilsvrc2014id2 = std::string(what[1].first, what[1].second);
								int localid = atol(std::string(what[2].first, what[2].second).c_str());
								localid_list.push_back(localid);
								++total_training_image_count;
							}
						}
					}
				}
			}
		}
	}
	std::cout << total_training_image_count << " training images found\n";
	std::map<std::string, std::pair<unsigned int, float> > ilsvrc2014id_to_localid_count_and_remaining_ratio_map;
	for(std::map<std::string, std::vector<unsigned int> >::iterator it = ilsvrc2014id_to_localid_list_map.begin(); it != ilsvrc2014id_to_localid_list_map.end(); ++it)
		ilsvrc2014id_to_localid_count_and_remaining_ratio_map.insert(std::make_pair(it->first, std::make_pair(it->second.size(), it->second.size() > 0 ? 1.0F : 0.0F)));
	nnforge::random_generator rnd;

	nnforge::varying_data_stream_writer_smart_ptr training_data_writer;
	{
		boost::filesystem::path training_file_path = get_working_data_folder() / training_randomized_data_filename;
		std::cout << "Writing randomized training data to " << training_file_path.string() << "..." << std::endl;
		nnforge_shared_ptr<std::ofstream> training_file(new boost::filesystem::ofstream(training_file_path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc));
		training_data_writer = nnforge::varying_data_stream_writer_smart_ptr(new nnforge::varying_data_stream_writer(
			training_file,
			total_training_image_count));
	}

	std::vector<std::string> best_ilsvrc2014id_list;
	for(unsigned int entry_to_write_count = 0; entry_to_write_count < total_training_image_count; ++entry_to_write_count)
	{
		if (best_ilsvrc2014id_list.empty())
		{
			float best_ratio = -1.0F;
			for(std::map<std::string, std::pair<unsigned int, float> >::const_iterator it = ilsvrc2014id_to_localid_count_and_remaining_ratio_map.begin(); it != ilsvrc2014id_to_localid_count_and_remaining_ratio_map.end(); ++it)
			{
				float new_ratio = it->second.second;
				if (new_ratio > best_ratio)
				{
					best_ilsvrc2014id_list.clear();
					best_ilsvrc2014id_list.push_back(it->first);
					best_ratio = new_ratio;
				}
				else if (new_ratio == best_ratio)
					best_ilsvrc2014id_list.push_back(it->first);
			}
		}

		std::string best_ilsvrc2014id;
		{
			nnforge_uniform_int_distribution<unsigned int> dist(0, static_cast<unsigned int>(best_ilsvrc2014id_list.size()) - 1);
			unsigned int index = dist(rnd);
			best_ilsvrc2014id = best_ilsvrc2014id_list[index];
			best_ilsvrc2014id_list[index] = best_ilsvrc2014id_list.back();
			best_ilsvrc2014id_list.pop_back();
		}

		std::map<std::string, std::vector<unsigned int> >::iterator bucket_it = ilsvrc2014id_to_localid_list_map.find(best_ilsvrc2014id);
		std::vector<unsigned int>& localid_list = bucket_it->second;
		if (localid_list.empty())
			throw std::runtime_error("Unexpected error in prepare_training_data: No elements left");

		nnforge_uniform_int_distribution<unsigned int> dist(0, static_cast<unsigned int>(localid_list.size()) - 1);

		unsigned int index = dist(rnd);
		unsigned int local_id = localid_list[index];
		unsigned int leftover_local_id = localid_list[localid_list.size() - 1];
		localid_list[index] = leftover_local_id;
		localid_list.pop_back();
		std::map<std::string, std::pair<unsigned int, float> >::iterator it = ilsvrc2014id_to_localid_count_and_remaining_ratio_map.find(best_ilsvrc2014id);
		it->second.second = static_cast<float>(localid_list.size()) / static_cast<float>(it->second.first);

		std::string filename = (boost::format("%1%_%2%.JPEG") % best_ilsvrc2014id % local_id).str();
		boost::filesystem::path image_file_path = training_images_folder_path / best_ilsvrc2014id / filename;
		int class_id = get_classid_by_wnid(get_wnid_by_ilsvrc2014id(best_ilsvrc2014id));

		write_supervised_data(image_file_path, *training_data_writer, class_id);

		if (((entry_to_write_count + 1) % 100000) == 0)
			std::cout << (entry_to_write_count + 1) << " entries written" << std::endl;
	}
	std::cout << total_training_image_count << " entries written" << std::endl;
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

	nnforge::random_generator rnd;

	nnforge::varying_data_stream_writer_smart_ptr training_data_writer;
	{
		boost::filesystem::path training_file_path = get_working_data_folder() / training_randomized_data_filename;
		std::cout << "Writing randomized training data to " << training_file_path.string() << "..." << std::endl;
		nnforge_shared_ptr<std::ofstream> training_file(new boost::filesystem::ofstream(training_file_path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc));
		training_data_writer = nnforge::varying_data_stream_writer_smart_ptr(new nnforge::varying_data_stream_writer(
			training_file,
			static_cast<unsigned int>(ilsvrc2014id_localid_pair_list.size())));
	}

	for(unsigned int entry_written_count = 0; entry_written_count < total_training_image_count; ++entry_written_count)
	{
		nnforge_uniform_int_distribution<unsigned int> dist(0, static_cast<unsigned int>(ilsvrc2014id_localid_pair_list.size()) - 1);

		unsigned int index = dist(rnd);
		std::pair<std::string, unsigned int> ilsvrc2014id_localid_pair = ilsvrc2014id_localid_pair_list[index];
		ilsvrc2014id_localid_pair_list[index] = ilsvrc2014id_localid_pair_list[ilsvrc2014id_localid_pair_list.size() - 1];
		ilsvrc2014id_localid_pair_list.pop_back();

		std::string filename = (boost::format("%1%_%2%.JPEG") % ilsvrc2014id_localid_pair.first % ilsvrc2014id_localid_pair.second).str();
		boost::filesystem::path image_file_path = training_images_folder_path / ilsvrc2014id_localid_pair.first / filename;
		int class_id = get_classid_by_wnid(get_wnid_by_ilsvrc2014id(ilsvrc2014id_localid_pair.first));

		write_supervised_data(image_file_path, *training_data_writer, class_id);

		if (((entry_written_count + 1) % 100000) == 0)
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

	nnforge::varying_data_stream_writer_smart_ptr validating_data_writer;
	{
		boost::filesystem::path validating_file_path = get_working_data_folder() / validating_data_filename;
		std::cout << "Writing validating data to " << validating_file_path.string() << "..." << std::endl;
		nnforge_shared_ptr<std::ofstream> validating_file(new boost::filesystem::ofstream(validating_file_path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc));
		validating_data_writer = nnforge::varying_data_stream_writer_smart_ptr(new nnforge::varying_data_stream_writer(
			validating_file,
			static_cast<unsigned int>(classid_list.size())));
	}

	boost::filesystem::path validating_images_folder_path = get_input_data_folder() / validating_images_folder_name;
	for(int i = 0; i < classid_list.size(); ++i)
	{
		unsigned int class_id = classid_list[i];
		unsigned int image_id = i + 1;
		boost::filesystem::path image_file_path = validating_images_folder_path / (boost::format("ILSVRC2012_val_%|1$08d|.JPEG") % image_id).str();

		write_supervised_data(image_file_path, *validating_data_writer, class_id);
	}
	std::cout << classid_list.size() << " entries written" << std::endl;
}

void imagenet_toolset::prepare_testing_data()
{
}

void imagenet_toolset::write_supervised_data(
	const boost::filesystem::path& image_file_path,
	nnforge::varying_data_stream_writer& writer,
	unsigned int class_id)
{
	uintmax_t file_size = boost::filesystem::file_size(image_file_path);
	std::vector<unsigned char> image_content(file_size + sizeof(unsigned int));
	{
		boost::filesystem::ifstream in(image_file_path, std::ios::binary);
		if (!in.read(reinterpret_cast<char *>(&(*image_content.begin())), file_size))
			throw std::runtime_error((boost::format("Error reading file %1%") % image_file_path.string()).str());
	}
	unsigned char * class_id_ptr = reinterpret_cast<unsigned char *>(&class_id);
	std::copy(class_id_ptr, class_id_ptr + sizeof(unsigned int), image_content.begin() + file_size);
	writer.raw_write(&(*image_content.begin()), image_content.size());
}

bool imagenet_toolset::is_training_with_validation() const
{
	return true;
}

std::vector<nnforge::data_transformer_smart_ptr> imagenet_toolset::get_input_data_transformer_list_for_training() const
{
	std::vector<nnforge::data_transformer_smart_ptr> res;
	
	res.push_back(nnforge::data_transformer_smart_ptr(new nnforge::intensity_2d_data_transformer(
		max_contrast_factor,
		max_brightness_shift)));
	res.push_back(nnforge::data_transformer_smart_ptr(new nnforge::distort_2d_data_transformer(
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
		
	res.push_back(nnforge::data_transformer_smart_ptr(new nnforge::convert_data_type_transformer()));
	res.push_back(get_input_data_normalize_transformer());

	res.push_back(nnforge::data_transformer_smart_ptr(new nnforge::uniform_intensity_data_transformer(
		std::vector<float>(3, -max_color_shift),
		std::vector<float>(3, max_color_shift))));

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

nnforge::supervised_data_reader_smart_ptr imagenet_toolset::get_initial_data_reader_for_normalizing() const
{
	nnforge_shared_ptr<std::istream> training_data_stream(new boost::filesystem::ifstream(get_working_data_folder() / training_randomized_data_filename, std::ios_base::in | std::ios_base::binary));
	nnforge::supervised_data_reader_smart_ptr current_reader(new nnforge::supervised_random_image_data_stream_reader(
		training_data_stream,
		training_image_original_width,
		training_image_original_height,
		training_image_width,
		training_image_height,
		class_count,
		true,
		true));
	current_reader = nnforge::supervised_data_reader_smart_ptr(new nnforge::supervised_transformed_input_data_reader(current_reader, nnforge::data_transformer_smart_ptr(new nnforge::convert_data_type_transformer())));
	return current_reader;
}

nnforge::supervised_data_reader_smart_ptr imagenet_toolset::get_initial_data_reader_for_training(bool force_deterministic) const
{
	nnforge_shared_ptr<std::istream> training_data_stream(new boost::filesystem::ifstream(get_working_data_folder() / training_randomized_data_filename, std::ios_base::in | std::ios_base::binary));

	nnforge::supervised_data_reader_smart_ptr res(new nnforge::supervised_random_image_data_stream_reader(
		training_data_stream,
		training_image_original_width,
		training_image_original_height,
		training_image_width,
		training_image_height,
		class_count,
		true,
		force_deterministic));

	return res;
}

nnforge::supervised_data_reader_smart_ptr imagenet_toolset::get_initial_data_reader_for_validating() const
{
	nnforge_shared_ptr<std::istream> validating_data_stream(new boost::filesystem::ifstream(get_working_data_folder() / validating_data_filename, std::ios_base::in | std::ios_base::binary));
	return get_validating_reader(validating_data_stream, rich_inference);
}

unsigned int imagenet_toolset::get_classifier_visualizer_top_n() const
{
	return 5;
}

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

std::vector<nnforge::bool_option> imagenet_toolset::get_bool_options()
{
	std::vector<nnforge::bool_option> res;

	res.push_back(nnforge::bool_option("rich_inference", &rich_inference, false, "Run multiple samples for each entry and average results"));

	return res;
}
