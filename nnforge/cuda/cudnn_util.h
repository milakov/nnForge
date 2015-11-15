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

#include <cudnn.h>

#include "../layer_configuration_specific.h"

namespace nnforge
{
	namespace cuda
	{
		class cudnn_util
		{
		public:
			static void set_tensor_descriptor(
				cudnnTensorDescriptor_t tensor_desc,
				const layer_configuration_specific& config,
				unsigned int entry_count);

			static void set_tensor_bias_descriptor(
				cudnnTensorDescriptor_t tensor_desc,
				unsigned int feature_map_count,
				unsigned int dimension_count);

			static void set_convolution_descriptor(
				cudnnConvolutionDescriptor_t convolution_desc,
				const std::vector<unsigned int> zero_padding);

			static void set_pooling_descriptor(
				cudnnPoolingDescriptor_t pooling_desc,
				cudnnPoolingMode_t pooling_mode,
				const std::vector<unsigned int>& subsampling_sizes);

			static void set_filter_descriptor(
				cudnnFilterDescriptor_t filter_desc,
				unsigned int output_feature_map_count,
				unsigned int input_feature_map_count,
				const std::vector<unsigned int>& windows_sizes);

		private:
			cudnn_util();
			cudnn_util(const cudnn_util&);
			cudnn_util& operator =(const cudnn_util&);
			~cudnn_util();
		};
	}
}
