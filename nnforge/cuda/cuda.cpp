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

#include "cuda.h"

#include "../nnforge.h"

#include "layer_testing_schema_factory.h"
#include "local_contrast_subtractive_layer_testing_schema.h"
#include "absolute_layer_testing_schema.h"
#include "hyperbolic_tangent_layer_testing_schema.h"
#include "average_subsampling_layer_testing_schema.h"
#include "convolution_layer_testing_schema.h"
#include "sparse_convolution_layer_testing_schema.h"
#include "max_subsampling_layer_testing_schema.h"
#include "rectified_linear_layer_testing_schema.h"
#include "softmax_layer_testing_schema.h"
#include "rgb_to_yuv_convert_layer_testing_schema.h"
#include "maxout_layer_testing_schema.h"
#include "sigmoid_layer_testing_schema.h"
#include "dropout_layer_testing_schema.h"
#include "parametric_rectified_linear_layer_testing_schema.h"
#include "untile_layer_testing_schema.h"
#include "lerror_layer_testing_schema.h"
#include "accuracy_layer_testing_schema.h"
#include "negative_log_likelihood_layer_testing_schema.h"
#include "cross_entropy_layer_testing_schema.h"
#include "gradient_modifier_layer_testing_schema.h"
#include "concat_layer_testing_schema.h"
#include "reshape_layer_testing_schema.h"
#include "cdf_max_layer_testing_schema.h"
#include "prefix_sum_layer_testing_schema.h"
#include "upsampling_layer_testing_schema.h"
#include "add_layer_testing_schema.h"
#include "cdf_to_pdf_layer_testing_schema.h"
#include "entry_convolution_layer_testing_schema.h"
#include "batch_norm_layer_testing_schema.h"

#include "layer_updater_schema_factory.h"
#include "rectified_linear_layer_updater_schema.h"
#include "sigmoid_layer_updater_schema.h"
#include "softmax_layer_updater_schema.h"
#include "hyperbolic_tangent_layer_updater_schema.h"
#include "maxout_layer_updater_schema.h"
#include "local_contrast_subtractive_layer_updater_schema.h"
#include "absolute_layer_updater_schema.h"
#include "parametric_rectified_linear_layer_updater_schema.h"
#include "dropout_layer_updater_schema.h"
#include "rgb_to_yuv_convert_layer_updater_schema.h"
#include "average_subsampling_layer_updater_schema.h"
#include "max_subsampling_layer_updater_schema.h"
#include "convolution_layer_updater_schema.h"
#include "sparse_convolution_layer_updater_schema.h"
#include "lerror_layer_updater_schema.h"
#include "accuracy_layer_updater_schema.h"
#include "negative_log_likelihood_layer_updater_schema.h"
#include "cross_entropy_layer_updater_schema.h"
#include "gradient_modifier_layer_updater_schema.h"
#include "concat_layer_updater_schema.h"
#include "reshape_layer_updater_schema.h"
#include "cdf_max_layer_updater_schema.h"
#include "prefix_sum_layer_updater_schema.h"
#include "upsampling_layer_updater_schema.h"
#include "add_layer_updater_schema.h"
#include "cdf_to_pdf_layer_updater_schema.h"
#include "entry_convolution_layer_updater_schema.h"
#include "batch_norm_layer_updater_schema.h"

namespace nnforge
{
	namespace cuda
	{
		void cuda::init()
		{
			nnforge::init();

			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new local_contrast_subtractive_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new absolute_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new hyperbolic_tangent_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new average_subsampling_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new convolution_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new sparse_convolution_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new max_subsampling_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new rectified_linear_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new softmax_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new rgb_to_yuv_convert_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new maxout_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new sigmoid_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new dropout_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new parametric_rectified_linear_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new untile_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new lerror_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new accuracy_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new negative_log_likelihood_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new cross_entropy_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new gradient_modifier_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new concat_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new reshape_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new cdf_max_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new prefix_sum_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new upsampling_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new add_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new cdf_to_pdf_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new entry_convolution_layer_testing_schema()));
			layer_testing_schema_factory::singleton::get_mutable_instance().register_layer_testing_schema(layer_testing_schema::ptr(new batch_norm_layer_testing_schema()));

			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new rectified_linear_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new sigmoid_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new softmax_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new hyperbolic_tangent_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new maxout_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new local_contrast_subtractive_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new absolute_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new parametric_rectified_linear_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new dropout_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new rgb_to_yuv_convert_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new average_subsampling_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new max_subsampling_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new convolution_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new sparse_convolution_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new lerror_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new accuracy_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new negative_log_likelihood_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new cross_entropy_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new gradient_modifier_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new concat_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new reshape_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new cdf_max_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new prefix_sum_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new upsampling_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new add_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new cdf_to_pdf_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new entry_convolution_layer_updater_schema()));
			layer_updater_schema_factory::singleton::get_mutable_instance().register_layer_updater_schema(layer_updater_schema::ptr(new batch_norm_layer_updater_schema()));
		}
	}
}
