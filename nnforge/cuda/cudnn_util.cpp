/*
 *  Copyright 2011-2017 Maxim Milakov
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
#include <algorithm>
#include <vector>

namespace nnforge
{
	namespace cuda
	{
		void cudnn_util::set_tensor_descriptor(
			cudnnTensorDescriptor_t tensor_desc,
			const layer_configuration_specific& config,
			unsigned int entry_count,
			const std::vector<unsigned int>& strides)
		{
			std::vector<int> tensor_dimensions(config.dimension_sizes.size() + 2);
			tensor_dimensions[0] = entry_count;
			tensor_dimensions[1] = config.feature_map_count;
			for(int i = 0; i < config.dimension_sizes.size(); ++i)
				tensor_dimensions[i + 2] = config.dimension_sizes[config.dimension_sizes.size() - 1 - i];
			for(int i = static_cast<int>(tensor_dimensions.size()); i < 4; ++i)
				tensor_dimensions.push_back(1);

			std::vector<int> tensor_strides(tensor_dimensions.size());
			std::vector<unsigned int> current_strides(strides);
			if (current_strides.empty())
				current_strides.resize(1, 1);
			std::copy(current_strides.begin(), current_strides.end(), tensor_strides.rbegin());
			for(int i = static_cast<int>(tensor_strides.size() - current_strides.size() - 1); i >= 0; --i)
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

		void cudnn_util::set_tensor_bn_weights_descriptor(
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
			const std::vector<unsigned int>& strides,
			const std::vector<unsigned int>& dilation)
		{
			std::vector<int> conv_padding(zero_padding.rbegin(), zero_padding.rend());
			std::vector<int> filter_stride(strides.rbegin(), strides.rend());
			std::vector<int> conv_dilation(dilation.rbegin(), dilation.rend());
			cudnn_safe_call(cudnnSetConvolutionNdDescriptor(
				convolution_desc,
				static_cast<int>(zero_padding.size()),
				&conv_padding[0],
				&filter_stride[0],
				&conv_dilation[0],
				CUDNN_CROSS_CORRELATION,
				CUDNN_DATA_FLOAT));
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
				CUDNN_TENSOR_NCHW,
				static_cast<int>(filter_dimensions.size()),
				&filter_dimensions[0]));
		}

		bool cudnn_util::is_over_sol_algos_available(
			const std::vector<unsigned int>& window_sizes,
			const std::vector<unsigned int>& strides,
			const std::vector<unsigned int>& dilation)
		{
			bool unit_stride = true;
			for(int i = 0; i < strides.size(); ++i)
				unit_stride = unit_stride && (strides[i] == 1);
			bool non_unit_window_size = false;
			for(int i = 0; i < window_sizes.size(); ++i)
				non_unit_window_size = non_unit_window_size || (window_sizes[i] > 1);
			bool unit_dilation = true;
			for(int i = 0; i < dilation.size(); ++i)
				unit_dilation = unit_dilation && (dilation[i] == 1);

			return (unit_stride && non_unit_window_size && unit_dilation);
		}

		tensor_params cudnn_util::get_tensor_params(cudnnTensorDescriptor_t tensor_desc)
		{
			tensor_params res;
			res.dims.resize(10);
			res.strides.resize(res.dims.size());
			int actual_dims;
			cudnn_safe_call(cudnnGetTensorNdDescriptor(
				tensor_desc,
				static_cast<int>(res.dims.size()),
				&res.data_type,
				&actual_dims,
				&res.dims[0],
				&res.strides[0]));
			res.dims.resize(actual_dims);
			res.strides.resize(actual_dims);

			return res;
		}

		filter_params cudnn_util::get_filter_params(cudnnFilterDescriptor_t filter_desc)
		{
			filter_params res;
			res.dims.resize(7);
			int actual_dims;
			cudnn_safe_call(cudnnGetFilterNdDescriptor(
				filter_desc,
				static_cast<int>(res.dims.size()),
				&res.data_type,
				&res.format,
				&actual_dims,
				&res.dims[0]));
			res.dims.resize(actual_dims);

			return res;
		}

		convolution_params cudnn_util::get_convolution_params(cudnnConvolutionDescriptor_t convolution_desc)
		{
			convolution_params res;
			res.padding.resize(5);
			res.strides.resize(res.padding.size());
			res.dilation.resize(res.padding.size());
			int actual_dims;
			cudnn_safe_call(cudnnGetConvolutionNdDescriptor(
				convolution_desc,
				static_cast<int>(res.padding.size()),
				&actual_dims,
				&res.padding[0],
				&res.strides[0],
				&res.dilation[0],
				&res.mode,
				&res.data_type));
			res.padding.resize(actual_dims);
			res.strides.resize(actual_dims);
			res.dilation.resize(actual_dims);

			return res;
		}

		bool operator<(const tensor_params&x, const tensor_params&y)
		{
			for(int i = 0; i < std::min(x.dims.size(), y.dims.size()); ++i)
			{
				if (x.dims[i] < y.dims[i])
					return true;
				else if (y.dims[i] < x.dims[i])
					return false;
			}

			for(int i = 0; i < std::min(x.strides.size(), y.strides.size()); ++i)
			{
				if (x.strides[i] < y.strides[i])
					return true;
				else if (y.strides[i] < x.strides[i])
					return false;
			}

			if (x.data_type < y.data_type)
				return true;
			else if (y.data_type < x.data_type)
				return false;

			return false;
		}

		bool operator<(const filter_params&x, const filter_params&y)
		{
			for(int i = 0; i < std::min(x.dims.size(), y.dims.size()); ++i)
			{
				if (x.dims[i] < y.dims[i])
					return true;
				else if (y.dims[i] < x.dims[i])
					return false;
			}

			if (x.data_type < y.data_type)
				return true;
			else if (y.data_type < x.data_type)
				return false;

			if (x.format < y.format)
				return true;
			else if (y.format < x.format)
				return false;

			return false;
		}

		bool operator<(const convolution_params&x, const convolution_params&y)
		{
			for(int i = 0; i < std::min(x.padding.size(), y.padding.size()); ++i)
			{
				if (x.padding[i] < y.padding[i])
					return true;
				else if (y.padding[i] < x.padding[i])
					return false;
			}

			for(int i = 0; i < std::min(x.strides.size(), y.strides.size()); ++i)
			{
				if (x.strides[i] < y.strides[i])
					return true;
				else if (y.strides[i] < x.strides[i])
					return false;
			}

			for(int i = 0; i < std::min(x.dilation.size(), y.dilation.size()); ++i)
			{
				if (x.dilation[i] < y.dilation[i])
					return true;
				else if (y.dilation[i] < x.dilation[i])
					return false;
			}

			if (x.data_type < y.data_type)
				return true;
			else if (y.data_type < x.data_type)
				return false;

			if (x.mode < y.mode)
				return true;
			else if (y.mode < x.mode)
				return false;

			return false;
		}
	}
}
	