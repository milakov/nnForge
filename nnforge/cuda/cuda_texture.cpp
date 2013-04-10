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

#include "cuda_texture.h"

#include "neural_network_cuda_exception.h"

#include <cstring>

namespace nnforge
{
	namespace cuda
	{
		cuda_texture::cuda_texture(const_cuda_linear_buffer_device_smart_ptr dev_smart_ptr)
			: tex(0)
		{
			struct cudaResourceDesc res_desc;
			memset(&res_desc, 0, sizeof(res_desc));
			res_desc.resType = cudaResourceTypeLinear;
			res_desc.res.linear.devPtr = const_cast<void *>((const void *)(*dev_smart_ptr));
			res_desc.res.linear.desc = cudaCreateChannelDesc<float>();
			res_desc.res.linear.sizeInBytes = dev_smart_ptr->get_size();

			struct cudaTextureDesc tex_desc;
			memset(&tex_desc, 0, sizeof(tex_desc));
			tex_desc.addressMode[0] = cudaAddressModeBorder;
			tex_desc.readMode = cudaReadModeElementType;
			tex_desc.normalizedCoords = 0;

			cuda_safe_call(cudaCreateTextureObject(&tex, &res_desc, &tex_desc, 0));
		}

		cuda_texture::~cuda_texture()
		{
			if (tex != 0)
				cudaDestroyTextureObject(tex);
		}

		cuda_texture::operator cudaTextureObject_t () const
		{
			return tex;
		}

		size_t cuda_texture::get_size() const
		{
			struct cudaResourceDesc res_desc;
			cuda_safe_call(cudaGetTextureObjectResourceDesc(&res_desc, tex));

			return res_desc.res.linear.sizeInBytes;
		}
	}
}
