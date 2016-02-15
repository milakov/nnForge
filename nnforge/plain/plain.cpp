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

#include "plain.h"

#include "../nnforge.h"

#include "layer_tester_plain_factory.h"

#include "absolute_layer_tester_plain.h"
#include "dropout_layer_tester_plain.h"
#include "hyperbolic_tangent_layer_tester_plain.h"
#include "rectified_linear_layer_tester_plain.h"
#include "rgb_to_yuv_convert_layer_tester_plain.h"
#include "sigmoid_layer_tester_plain.h"
#include "parametric_rectified_linear_layer_tester_plain.h"
#include "maxout_layer_tester_plain.h"
#include "average_subsampling_layer_tester_plain.h"
#include "max_subsampling_layer_tester_plain.h"
#include "untile_layer_tester_plain.h"
#include "softmax_layer_tester_plain.h"
#include "convolution_layer_tester_plain.h"
#include "sparse_convolution_layer_tester_plain.h"
#include "local_contrast_subtractive_layer_tester_plain.h"
#include "lerror_layer_tester_plain.h"
#include "cross_entropy_layer_tester_plain.h"
#include "negative_log_likelihood_layer_tester_plain.h"
#include "accuracy_layer_tester_plain.h"
#include "gradient_modifier_layer_tester_plain.h"
#include "concat_layer_tester_plain.h"
#include "reshape_layer_tester_plain.h"
#include "cdf_max_layer_tester_plain.h"
#include "prefix_sum_layer_tester_plain.h"
#include "upsampling_layer_tester_plain.h"
#include "add_layer_tester_plain.h"
#include "cdf_to_pdf_layer_tester_plain.h"
#include "entry_convolution_layer_tester_plain.h"

#include "layer_updater_plain_factory.h"

#include "hyperbolic_tangent_layer_updater_plain.h"
#include "sigmoid_layer_updater_plain.h"
#include "average_subsampling_layer_updater_plain.h"
#include "max_subsampling_layer_updater_plain.h"
#include "absolute_layer_updater_plain.h"
#include "rectified_linear_layer_updater_plain.h"
#include "maxout_layer_updater_plain.h"
#include "local_contrast_subtractive_layer_updater_plain.h"
#include "dropout_layer_updater_plain.h"
#include "softmax_layer_updater_plain.h"
#include "parametric_rectified_linear_layer_updater_plain.h"
#include "convolution_layer_updater_plain.h"
#include "sparse_convolution_layer_updater_plain.h"
#include "lerror_layer_updater_plain.h"
#include "cross_entropy_layer_updater_plain.h"
#include "negative_log_likelihood_layer_updater_plain.h"
#include "accuracy_layer_updater_plain.h"
#include "gradient_modifier_layer_updater_plain.h"
#include "concat_layer_updater_plain.h"
#include "reshape_layer_updater_plain.h"
#include "cdf_max_layer_updater_plain.h"
#include "prefix_sum_layer_updater_plain.h"
#include "upsampling_layer_updater_plain.h"
#include "add_layer_updater_plain.h"
#include "cdf_to_pdf_layer_updater_plain.h"
#include "entry_convolution_layer_updater_plain.h"

namespace nnforge
{
	namespace plain
	{
		void plain::init()
		{
			nnforge::init();

			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new absolute_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new dropout_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new hyperbolic_tangent_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new rectified_linear_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new rgb_to_yuv_convert_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new sigmoid_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new parametric_rectified_linear_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new maxout_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new average_subsampling_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new max_subsampling_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new untile_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new softmax_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new convolution_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new sparse_convolution_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new local_contrast_subtractive_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new lerror_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new cross_entropy_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new negative_log_likelihood_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new accuracy_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new gradient_modifier_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new concat_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new reshape_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new cdf_max_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new prefix_sum_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new upsampling_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new add_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new cdf_to_pdf_layer_tester_plain()));
			layer_tester_plain_factory::singleton::get_mutable_instance().register_layer_tester_plain(layer_tester_plain::ptr(new entry_convolution_layer_tester_plain()));

			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new hyperbolic_tangent_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new sigmoid_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new average_subsampling_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new max_subsampling_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new absolute_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new rectified_linear_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new maxout_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new local_contrast_subtractive_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new dropout_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new softmax_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new parametric_rectified_linear_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new convolution_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new sparse_convolution_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new lerror_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new cross_entropy_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new negative_log_likelihood_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new accuracy_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new gradient_modifier_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new concat_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new reshape_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new cdf_max_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new prefix_sum_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new upsampling_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new add_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new cdf_to_pdf_layer_updater_plain()));
			layer_updater_plain_factory::singleton::get_mutable_instance().register_layer_updater_plain(layer_updater_plain::ptr(new entry_convolution_layer_updater_plain()));
		}
	}
}
