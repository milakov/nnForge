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

#include "cudnn_util.h"

#include "neural_network_cudnn_exception.h"
#include <vector>

namespace nnforge
{
	namespace cuda
	{
		void cudnn_util::set_tensor_descriptor(
			cudnnTensorDescriptor_t tensor_desc,
			const layer_configuration_specific& config,
			unsigned int entry_count)
		{
			std::vector<int> tensor_dimensions(config.dimension_sizes.size() + 2);
			tensor_dimensions[0] = entry_count;
			tensor_dimensions[1] = config.feature_map_count;
			for(int i = 0; i < config.dimension_sizes.size(); ++i)
				tensor_dimensions[i + 2] = config.dimension_sizes[config.dimension_sizes.size() - 1 - i];
			for(int i = static_cast<int>(tensor_dimensions.size()); i < 4; ++i)
				tensor_dimensions.push_back(1);

			std::vector<int> tensor_strides(tensor_dimensions.size());
			tensor_strides.back() = 1;
			for(int i = static_cast<int>(tensor_strides.size()) - 2; i >= 0; --i)
				tensor_strides[i] = tensor_strides[i + 1] * tensor_dimensions[i + 1];

			cudnn_safe_call(cudnnSetTensorNdDescriptor(
				tensor_desc,
				CUDNN_DATA_FLOAT,
				static_cast<int>(tensor_dimensions.size()),
				&tensor_dimensions[0],
				&tensor_strides[0]));
		}
	
		void cudnn_util::set_tensor_bias_descriptor(
			cudnnTensorDescriptor_t tensor_desc,
			unsigned int feature_map_count,
			unsigned int dimension_count)
		{
			cudnn_util::set_tensor_descriptor(
				tensor_desc,
				layer_configuration_specific(feature_map_count, std::vector<unsigned int>(dimension_count, 1)),
				1);
		}

		void cudnn_util::set_convolution_descriptor(
			cudnnConvolutionDescriptor_t convolution_desc,
			const std::vector<unsigned int>& zero_padding,
			const std::vector<unsigned int>& strides)
		{
			std::vector<int> conv_padding(zero_padding.rbegin(), zero_padding.rend());
			std::vector<int> filter_stride(strides.rbegin(), strides.rend());
			std::vector<int> upscale(zero_padding.size(), 1);
			cudnn_safe_call(cudnnSetConvolutionNdDescriptor_v3(
				convolution_desc,
				static_cast<int>(zero_padding.size()),
				&conv_padding[0],
				&filter_stride[0],
				&upscale[0],
				CUDNN_CROSS_CORRELATION,
				CUDNN_DATA_FLOAT));
		}

		void cudnn_util::set_pooling_descriptor(
			cudnnPoolingDescriptor_t pooling_desc,
			cudnnPoolingMode_t pooling_mode,
			const std::vector<unsigned int>& subsampling_sizes)
		{
			std::vector<int> padding(subsampling_sizes.size(), 0);
			std::vector<int> dimensions(subsampling_sizes.rbegin(), subsampling_sizes.rend());
			cudnn_safe_call(cudnnSetPoolingNdDescriptor(
				pooling_desc,
				pooling_mode,
				static_cast<int>(subsampling_sizes.size()),
				&dimensions[0],
				&padding[0],
				&dimensions[0]));
		}

		void cudnn_util::set_filter_descriptor(
			cudnnFilterDescriptor_t filter_desc,
			unsigned int output_feature_map_count,
			unsigned int input_feature_map_count,
			const std::vector<unsigned int>& windows_sizes)
		{
			std::vector<int> filter_dimensions(windows_sizes.size() + 2);
			filter_dimensions[0] = output_feature_map_count;
			filter_dimensions[1] = input_feature_map_count;
			for(int i = 0; i < windows_sizes.size(); ++i)
				filter_dimensions[i + 2] = windows_sizes[windows_sizes.size() - 1 - i];

			cudnn_safe_call(cudnnSetFilterNdDescriptor(
				filter_desc,
				CUDNN_DATA_FLOAT,
				static_cast<int>(filter_dimensions.size()),
				&filter_dimensions[0]));
		}

		bool cudnn_util::is_over_sol_algos_available(
			const std::vector<unsigned int>& window_sizes,
			const std::vector<unsigned int>& strides)
		{
			bool unit_stride = true;
			for(int i = 0; i < strides.size(); ++i)
				unit_stride = unit_stride && (strides[i] == 1);
			bool non_unit_window_size = false;
			for(int i = 0; i < window_sizes.size(); ++i)
				non_unit_window_size = non_unit_window_size || (window_sizes[i] > 1);

			return (unit_stride && non_unit_window_size);
		}
	}
}
	