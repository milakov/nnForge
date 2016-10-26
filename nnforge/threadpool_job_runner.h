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

#include "nn_types.h"

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0501
#endif 

#include <boost/asio/io_service.hpp>

namespace nnforge
{
	// To launch job run service.post(boost::bind(function, arguments, ...));
	class threadpool_job_runner
	{
	public:
		typedef nnforge_shared_ptr<threadpool_job_runner> ptr;

		threadpool_job_runner(unsigned int thread_count);

		~threadpool_job_runner();

	public:
		unsigned int thread_count;
		boost::asio::io_service service;

	private:
		void * threadpool;
		boost::asio::io_service::work work;

	private:
		threadpool_job_runner();
		threadpool_job_runner(const threadpool_job_runner&);
		threadpool_job_runner& operator =(const threadpool_job_runner&);
	};
}
