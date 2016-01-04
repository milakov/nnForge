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

#include "nnforge.h"

#include "layer_factory.h"

#include <google/protobuf/stubs/common.h>

namespace nnforge
{
	void nnforge::init()
	{
		GOOGLE_PROTOBUF_VERIFY_VERSION;

		single_layer_factory::get_mutable_instance().register_layer(layer::ptr(new convolution_layer(std::vector<unsigned int>(1, 1), 1, 1)));
		single_layer_factory::get_mutable_instance().register_layer(layer::ptr(new sparse_convolution_layer(std::vector<unsigned int>(1, 1), 1, 1, 1U)));
		single_layer_factory::get_mutable_instance().register_layer(layer::ptr(new hyperbolic_tangent_layer()));
		single_layer_factory::get_mutable_instance().register_layer(layer::ptr(new average_subsampling_layer(std::vector<unsigned int>(1, 1))));
		single_layer_factory::get_mutable_instance().register_layer(layer::ptr(new max_subsampling_layer(std::vector<unsigned int>(1, 1))));
		single_layer_factory::get_mutable_instance().register_layer(layer::ptr(new absolute_layer()));
		single_layer_factory::get_mutable_instance().register_layer(layer::ptr(new local_contrast_subtractive_layer(std::vector<unsigned int>(1, 1), std::vector<unsigned int>(1, 0), 1)));
		single_layer_factory::get_mutable_instance().register_layer(layer::ptr(new rgb_to_yuv_convert_layer(std::vector<color_feature_map_config>(1, color_feature_map_config(0, 1, 2)))));
		single_layer_factory::get_mutable_instance().register_layer(layer::ptr(new rectified_linear_layer()));
		single_layer_factory::get_mutable_instance().register_layer(layer::ptr(new softmax_layer()));
		single_layer_factory::get_mutable_instance().register_layer(layer::ptr(new maxout_layer(2)));
		single_layer_factory::get_mutable_instance().register_layer(layer::ptr(new sigmoid_layer()));
		single_layer_factory::get_mutable_instance().register_layer(layer::ptr(new dropout_layer(0.5F)));
		single_layer_factory::get_mutable_instance().register_layer(layer::ptr(new parametric_rectified_linear_layer(1)));
		single_layer_factory::get_mutable_instance().register_layer(layer::ptr(new untile_layer(std::vector<std::vector<unsigned int> >(1, std::vector<unsigned int>(1, 2)))));
		single_layer_factory::get_mutable_instance().register_layer(layer::ptr(new data_layer()));
		single_layer_factory::get_mutable_instance().register_layer(layer::ptr(new lerror_layer()));
		single_layer_factory::get_mutable_instance().register_layer(layer::ptr(new accuracy_layer()));
		single_layer_factory::get_mutable_instance().register_layer(layer::ptr(new negative_log_likelihood_layer()));
		single_layer_factory::get_mutable_instance().register_layer(layer::ptr(new cross_entropy_layer()));
		single_layer_factory::get_mutable_instance().register_layer(layer::ptr(new gradient_modifier_layer()));
		single_layer_factory::get_mutable_instance().register_layer(layer::ptr(new concat_layer()));
		single_layer_factory::get_mutable_instance().register_layer(layer::ptr(new reshape_layer()));
	}
}
