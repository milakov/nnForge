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

#include <nnforge/nnforge.h>
#include <opencv2/core/core.hpp>
#include <map>
#include <boost/filesystem.hpp>

#include <nnforge/toolset.h>

class gtsrb_toolset : public nnforge::toolset
{
public:
	gtsrb_toolset(nnforge::factory_generator::ptr factory);

	virtual ~gtsrb_toolset();

protected:
	virtual void prepare_training_data();

	void write_single_entry(
		nnforge::structured_data_stream_writer& image_writer,
		nnforge::structured_data_stream_writer& label_writer,
		const boost::filesystem::path& absolute_file_path,
		unsigned int class_id,
		unsigned int roi_top_left_x,
		unsigned int roi_top_left_y,
		unsigned int roi_bottom_right_x,
		unsigned int roi_bottom_right_y,
		float rotation_angle_in_degrees = 0.0F,
		float scale_factor = 1.0F,
		float shift_x = 0.0F,
		float shift_y = 0.0F,
		float contrast = 1.0F,
		float brightness_shift = 0.0F);

	void write_folder(
		nnforge::structured_data_stream_writer& image_writer,
		nnforge::structured_data_stream_writer& label_writer,
		const boost::filesystem::path& relative_subfolder_path,
		const char * annotation_file_name,
		bool jitter);

	static const unsigned int image_width;
	static const unsigned int image_height;
	static const unsigned int class_count;
	static const bool use_roi;
	static const float max_rotation_angle_in_degrees;
	static const float max_scale_factor;
	static const float max_shift;
	static const float max_contrast_factor;
	static const float max_brightness_shift;
	static const unsigned int random_sample_count;
};
