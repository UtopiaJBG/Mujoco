// Copyright 2023 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MUJOCO_INCLUDE_MJTHREAD_H_
#define MUJOCO_INCLUDE_MJTHREAD_H_

#define mjMAXTHREAD 128        // maximum number of threads in a thread pool

typedef enum mjtTaskStatus_ {  // status values for mjTask
  mjTASK_NEW = 0,              // newly created
  mjTASK_QUEUED,               // enqueued in a thread pool
  mjTASK_COMPLETED             // completed execution
} mjtTaskStatus;

// function pointer type for mjTask
typedef void* (*mjfTask)(void*);

// An opaque type representing a thread pool.
struct mjThreadPool_ {
  int nworker;  // number of workers in the pool
};
typedef struct mjThreadPool_ mjThreadPool;

struct mjTask_ {        // a task that can be executed by a thread pool.
  mjfTask func;         // pointer to the function that implements the task
  void* args;           // arguments to func
  volatile int status;  // status of the task
};
typedef struct mjTask_ mjTask;

#endif  // MUJOCO_INCLUDE_MJTHREAD_H_
