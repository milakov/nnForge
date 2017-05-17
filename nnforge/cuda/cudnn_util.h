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

#include <cudnn.h>

#include "../layer_configuration_specific.h"

namespace nnforge
{
	namespace cuda
	{
		struct tensor_params
		{
			cudnnDataType_t data_type;
			std::vector<int> dims;
			std::vector<int> strides;
		};

		bool operator<(const tensor_params&x, const tensor_params&y);

		struct filter_params
		{
			cudnnDataType_t data_type;
			cudnnTensorFormat_t format;
			std::vector<int> dims;
		};

		bool operator<(const filter_params&x, const filter_params&y);

		struct convolution_params
		{
			cudnnConvolutionMode_t mode;
			cudnnDataType_t data_type;
			std::vector<int> padding;
			std::vector<int> strides;
			std::vector<int> dilation;
		};

		bool operator<(const convolution_params&x, const convolution_params&y);

		class cudnn_util
		{
		public:
			static void set_tensor_descriptor(
				cudnnTensorDescriptor_t tensor_desc,
				const layer_configuration_specific& config,
				unsigned int entry_count,
				const std::vector<unsigned int>& strides = std::vector<unsigned int>());

			static tensor_params get_tensor_params(cudnnTensorDescriptor_t tensor_desc);

			static void set_tensor_bias_descriptor(
				cudnnTensorDescriptor_t tensor_desc,
				unsigned int feature_map_count,
				unsigned int dimension_count);

			static void set_tensor_bn_weights_descriptor(
				cudnnTensorDescriptor_t tensor_desc,
				unsigned int feature_map_count,
				unsigned int dimension_count);

			static void set_convolution_descriptor(
				cudnnConvolutionDescriptor_t convolution_desc,
				const std::vector<unsigned int>& zero_padding,
				const std::vector<unsigned int>& strides,
				const std::vector<unsigned int>& dilation);

			static convolution_params get_convolution_params(cudnnConvolutionDescriptor_t convolution_desc);

			static void set_filter_descriptor(
				cudnnFilterDescriptor_t filter_desc,
				unsigned int output_feature_map_count,
				unsigned int input_feature_map_count,
				const std::vector<unsigned int>& windows_sizes);

			static filter_params get_filter_params(cudnnFilterDescriptor_t filter_desc);

			static bool is_over_sol_algos_available(
				const std::vector<unsigned int>& window_sizes,
				const std::vector<unsigned int>& strides,
				const std::vector<unsigned int>& dilation);

		private:
			cudnn_util() = delete;
			cudnn_util(const cudnn_util&) = delete;
			cudnn_util& operator =(const cudnn_util&) = delete;
			~cudnn_util() = delete;
		};
	}
}
