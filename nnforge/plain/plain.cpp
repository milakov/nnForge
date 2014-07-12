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

#include "plain.h"

#include "../nnforge.h"
#include "layer_tester_plain_factory.h"
#include "absolute_layer_tester_plain.h"
#include "hyperbolic_tangent_layer_tester_plain.h"
#include "average_subsampling_layer_tester_plain.h"
#include "max_subsampling_layer_tester_plain.h"
#include "local_contrast_subtractive_layer_tester_plain.h"
#include "convolution_layer_tester_plain.h"
#include "rectified_linear_layer_tester_plain.h"
#include "soft_rectified_linear_layer_tester_plain.h"
#include "softmax_layer_tester_plain.h"
#include "maxout_layer_tester_plain.h"
#include "rgb_to_yuv_convert_layer_tester_plain.h"
#include "sigmoid_layer_tester_plain.h"

#include "layer_hessian_plain_factory.h"
#include "absolute_layer_hessian_plain.h"
#include "hyperbolic_tangent_layer_hessian_plain.h"
#include "average_subsampling_layer_hessian_plain.h"
#include "max_subsampling_layer_hessian_plain.h"
#include "local_contrast_subtractive_layer_hessian_plain.h"
#include "convolution_layer_hessian_plain.h"
#include "rectified_linear_layer_hessian_plain.h"
#include "soft_rectified_linear_layer_hessian_plain.h"
#include "softmax_layer_hessian_plain.h"
#include "maxout_layer_hessian_plain.h"
#include "sigmoid_layer_hessian_plain.h"

#include "layer_updater_plain_factory.h"
#include "absolute_layer_updater_plain.h"
#include "hyperbolic_tangent_layer_updater_plain.h"
#include "average_subsampling_layer_updater_plain.h"
#include "max_subsampling_layer_updater_plain.h"
#include "local_contrast_subtractive_layer_updater_plain.h"
#include "convolution_layer_updater_plain.h"
#include "rectified_linear_layer_updater_plain.h"
#include "soft_rectified_linear_layer_updater_plain.h"
#include "softmax_layer_updater_plain.h"
#include "maxout_layer_updater_plain.h"
#include "sigmoid_layer_updater_plain.h"

namespace nnforge
{
	namespace plain
	{
		void plain::init()
		{
			nnforge::init();

			single_layer_tester_plain_factory::get_mutable_instance().register_layer_tester_plain(layer_tester_plain_smart_ptr(new absolute_layer_tester_plain()));
			single_layer_tester_plain_factory::get_mutable_instance().register_layer_tester_plain(layer_tester_plain_smart_ptr(new hyperbolic_tangent_layer_tester_plain()));
			single_layer_tester_plain_factory::get_mutable_instance().register_layer_tester_plain(layer_tester_plain_smart_ptr(new average_subsampling_layer_tester_plain()));
			single_layer_tester_plain_factory::get_mutable_instance().register_layer_tester_plain(layer_tester_plain_smart_ptr(new max_subsampling_layer_tester_plain()));
			single_layer_tester_plain_factory::get_mutable_instance().register_layer_tester_plain(layer_tester_plain_smart_ptr(new local_contrast_subtractive_layer_tester_plain()));
			single_layer_tester_plain_factory::get_mutable_instance().register_layer_tester_plain(layer_tester_plain_smart_ptr(new convolution_layer_tester_plain()));
			single_layer_tester_plain_factory::get_mutable_instance().register_layer_tester_plain(layer_tester_plain_smart_ptr(new rectified_linear_layer_tester_plain()));
			single_layer_tester_plain_factory::get_mutable_instance().register_layer_tester_plain(layer_tester_plain_smart_ptr(new soft_rectified_linear_layer_tester_plain()));
			single_layer_tester_plain_factory::get_mutable_instance().register_layer_tester_plain(layer_tester_plain_smart_ptr(new softmax_layer_tester_plain()));
			single_layer_tester_plain_factory::get_mutable_instance().register_layer_tester_plain(layer_tester_plain_smart_ptr(new maxout_layer_tester_plain()));
			single_layer_tester_plain_factory::get_mutable_instance().register_layer_tester_plain(layer_tester_plain_smart_ptr(new rgb_to_yuv_convert_layer_tester_plain()));
			single_layer_tester_plain_factory::get_mutable_instance().register_layer_tester_plain(layer_tester_plain_smart_ptr(new sigmoid_layer_tester_plain()));

			single_layer_hessian_plain_factory::get_mutable_instance().register_layer_hessian_plain(layer_hessian_plain_smart_ptr(new absolute_layer_hessian_plain()));
			single_layer_hessian_plain_factory::get_mutable_instance().register_layer_hessian_plain(layer_hessian_plain_smart_ptr(new hyperbolic_tangent_layer_hessian_plain()));
			single_layer_hessian_plain_factory::get_mutable_instance().register_layer_hessian_plain(layer_hessian_plain_smart_ptr(new average_subsampling_layer_hessian_plain()));
			single_layer_hessian_plain_factory::get_mutable_instance().register_layer_hessian_plain(layer_hessian_plain_smart_ptr(new max_subsampling_layer_hessian_plain()));
			single_layer_hessian_plain_factory::get_mutable_instance().register_layer_hessian_plain(layer_hessian_plain_smart_ptr(new local_contrast_subtractive_layer_hessian_plain()));
			single_layer_hessian_plain_factory::get_mutable_instance().register_layer_hessian_plain(layer_hessian_plain_smart_ptr(new convolution_layer_hessian_plain()));
			single_layer_hessian_plain_factory::get_mutable_instance().register_layer_hessian_plain(layer_hessian_plain_smart_ptr(new rectified_linear_layer_hessian_plain()));
			single_layer_hessian_plain_factory::get_mutable_instance().register_layer_hessian_plain(layer_hessian_plain_smart_ptr(new soft_rectified_linear_layer_hessian_plain()));
			single_layer_hessian_plain_factory::get_mutable_instance().register_layer_hessian_plain(layer_hessian_plain_smart_ptr(new softmax_layer_hessian_plain()));
			single_layer_hessian_plain_factory::get_mutable_instance().register_layer_hessian_plain(layer_hessian_plain_smart_ptr(new maxout_layer_hessian_plain()));
			single_layer_hessian_plain_factory::get_mutable_instance().register_layer_hessian_plain(layer_hessian_plain_smart_ptr(new sigmoid_layer_hessian_plain()));

			single_layer_updater_plain_factory::get_mutable_instance().register_layer_updater_plain(layer_updater_plain_smart_ptr(new absolute_layer_updater_plain()));
			single_layer_updater_plain_factory::get_mutable_instance().register_layer_updater_plain(layer_updater_plain_smart_ptr(new hyperbolic_tangent_layer_updater_plain()));
			single_layer_updater_plain_factory::get_mutable_instance().register_layer_updater_plain(layer_updater_plain_smart_ptr(new average_subsampling_layer_updater_plain()));
			single_layer_updater_plain_factory::get_mutable_instance().register_layer_updater_plain(layer_updater_plain_smart_ptr(new max_subsampling_layer_updater_plain()));
			single_layer_updater_plain_factory::get_mutable_instance().register_layer_updater_plain(layer_updater_plain_smart_ptr(new local_contrast_subtractive_layer_updater_plain()));
			single_layer_updater_plain_factory::get_mutable_instance().register_layer_updater_plain(layer_updater_plain_smart_ptr(new convolution_layer_updater_plain()));
			single_layer_updater_plain_factory::get_mutable_instance().register_layer_updater_plain(layer_updater_plain_smart_ptr(new rectified_linear_layer_updater_plain()));
			single_layer_updater_plain_factory::get_mutable_instance().register_layer_updater_plain(layer_updater_plain_smart_ptr(new soft_rectified_linear_layer_updater_plain()));
			single_layer_updater_plain_factory::get_mutable_instance().register_layer_updater_plain(layer_updater_plain_smart_ptr(new softmax_layer_updater_plain()));
			single_layer_updater_plain_factory::get_mutable_instance().register_layer_updater_plain(layer_updater_plain_smart_ptr(new maxout_layer_updater_plain()));
			single_layer_updater_plain_factory::get_mutable_instance().register_layer_updater_plain(layer_updater_plain_smart_ptr(new sigmoid_layer_updater_plain()));
		}
	}
}
