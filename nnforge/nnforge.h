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

#include "neural_network_exception.h"

#include "convolution_layer.h"
#include "sparse_convolution_layer.h"
#include "hyperbolic_tangent_layer.h"
#include "average_subsampling_layer.h"
#include "max_subsampling_layer.h"
#include "absolute_layer.h"
#include "local_contrast_subtractive_layer.h"
#include "rgb_to_yuv_convert_layer.h"
#include "rectified_linear_layer.h"
#include "softmax_layer.h"
#include "maxout_layer.h"
#include "sigmoid_layer.h"
#include "dropout_layer.h"
#include "parametric_rectified_linear_layer.h"
#include "untile_layer.h"
#include "data_layer.h"
#include "lerror_layer.h"
#include "accuracy_layer.h"
#include "negative_log_likelihood_layer.h"
#include "cross_entropy_layer.h"
#include "gradient_modifier_layer.h"
#include "concat_layer.h"
#include "reshape_layer.h"
#include "cdf_max_layer.h"
#include "prefix_sum_layer.h"
#include "upsampling_layer.h"
#include "add_layer.h"
#include "cdf_to_pdf_layer.h"
#include "entry_convolution_layer.h"
#include "batch_norm_layer.h"
#include "affine_grid_generator_layer.h"
#include "linear_sampler_layer.h"
#include "exponential_linear_layer.h"

#include "rnd.h"

#include "structured_data_stream_writer.h"
#include "varying_data_stream_reader.h"
#include "varying_data_stream_writer.h"
#include "structured_from_raw_data_reader.h"
#include "structured_data_bunch_mix_reader.h"
#include "neuron_value_set_data_bunch_reader.h"

#include "data_transformer_util.h"

#include "convert_to_polar_data_transformer.h"
#include "distort_2d_data_transformer.h"
#include "distort_2d_data_sampler_transformer.h"
#include "elastic_deformation_2d_data_transformer.h"
#include "embed_data_transformer.h"
#include "extract_data_transformer.h"
#include "intensity_2d_data_transformer.h"
#include "noise_data_transformer.h"
#include "reshape_data_transformer.h"
#include "rotate_band_data_transformer.h"
#include "uniform_intensity_data_transformer.h"
#include "normalize_data_transformer.h"
#include "natural_image_data_transformer.h"

namespace nnforge
{
	class nnforge
	{
	public:
		static void init();

	private:
		nnforge() = delete;
		~nnforge() = delete;
	};
}
