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

#include "threadpool_job_runner.h"

#include <boost/thread/thread.hpp>

namespace nnforge
{
	threadpool_job_runner::threadpool_job_runner(unsigned int thread_count)
		: thread_count(thread_count)
		, work(service)
		, threadpool(new boost::thread_group())
	{
		for(unsigned int i = 0; i < thread_count; ++i)
			((boost::thread_group *)threadpool)->create_thread(boost::bind(&boost::asio::io_service::run, &service));
	}

	threadpool_job_runner::~threadpool_job_runner()
	{
		service.stop();
		((boost::thread_group *)threadpool)->join_all();
		delete ((boost::thread_group *)threadpool);
	}
}
