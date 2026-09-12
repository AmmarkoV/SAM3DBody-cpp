#pragma once
// Windows implementation of the worker-pool protocol used by Pipeline::Impl.
// A generation counter makes both the initial wait and subsequent kicks safe
// against missed notifications and spurious condition-variable wakeups.
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

struct workerPool;
struct threadContext {
    void* argumentToPass = nullptr;
    workerPool* pool = nullptr;
    unsigned int threadID = 0;
    std::uint64_t generation = 0;
};

struct workerPool {
    std::mutex mutex;
    std::condition_variable start, complete;
    std::vector<threadContext> contexts;
    std::vector<std::thread> threads;
    unsigned int numberOfThreads = 0;
    unsigned int ready = 0, active = 0;
    std::uint64_t generation = 0;
    bool initialized = false, stopping = false;
};

static int threadpoolWorkerInitialWait(threadContext* ctx)
{
    auto& p = *ctx->pool;
    std::unique_lock<std::mutex> lock(p.mutex);
    ++p.ready;
    p.complete.notify_all();
    p.start.wait(lock, [&] { return p.stopping || p.generation != ctx->generation; });
    ctx->generation = p.generation;
    return !p.stopping;
}

static int threadpoolWorkerLoopCondition(threadContext* ctx)
{
    std::lock_guard<std::mutex> lock(ctx->pool->mutex);
    return !ctx->pool->stopping;
}

static int threadpoolWorkerLoopEnd(threadContext* ctx)
{
    auto& p = *ctx->pool;
    std::unique_lock<std::mutex> lock(p.mutex);
    if (p.active > 0) --p.active;
    p.complete.notify_all();
    p.start.wait(lock, [&] { return p.stopping || p.generation != ctx->generation; });
    ctx->generation = p.generation;
    return !p.stopping;
}

static int threadpoolMainThreadPrepareWorkForWorkers(workerPool* p)
{
    if (!p) return 0;
    std::lock_guard<std::mutex> lock(p->mutex);
    return p->initialized && !p->stopping && p->active == 0;
}

static int threadpoolMainThreadKickWorkers(workerPool* p)
{
    if (!p) return 0;
    std::lock_guard<std::mutex> lock(p->mutex);
    if (!p->initialized || p->stopping || p->active != 0) return 0;
    p->active = p->numberOfThreads;
    ++p->generation;
    p->start.notify_all();
    return 1;
}

static int threadpoolMainThreadWaitForKickedWorkersToFinishTimeoutSeconds(
    workerPool* p, unsigned int timeoutSeconds)
{
    if (!p) return 0;
    std::unique_lock<std::mutex> lock(p->mutex);
    if (!p->initialized) return 0;
    auto done = [&] { return p->active == 0 || p->stopping; };
    if (timeoutSeconds == 0) p->complete.wait(lock, done);
    else if (!p->complete.wait_for(lock, std::chrono::seconds(timeoutSeconds), done)) return 0;
    return !p->stopping;
}

static int threadpoolMainThreadWaitForWorkersToFinishTimeoutSeconds(
    workerPool* p, unsigned int timeoutSeconds)
{
    return threadpoolMainThreadKickWorkers(p) &&
           threadpoolMainThreadWaitForKickedWorkersToFinishTimeoutSeconds(p, timeoutSeconds);
}

static int threadpoolMainThreadWaitForWorkersToFinish(workerPool* p)
{
    return threadpoolMainThreadWaitForWorkersToFinishTimeoutSeconds(p, 0);
}

static int threadpoolDestroy(workerPool* p)
{
    if (!p) return 0;
    {
        std::lock_guard<std::mutex> lock(p->mutex);
        p->stopping = true;
    }
    p->start.notify_all();
    p->complete.notify_all();
    for (auto& t : p->threads) if (t.joinable()) t.join();
    p->threads.clear();
    p->contexts.clear();
    p->initialized = false;
    p->numberOfThreads = p->active = p->ready = 0;
    return 1;
}

static int threadpoolCreate(workerPool* p, unsigned int count,
                            void* workerFunction, void* argument)
{
    if (!p || p->initialized || !count || !workerFunction) return 0;
    p->stopping = false;
    p->generation = 0;
    p->ready = p->active = 0;
    p->numberOfThreads = count;
    try {
        p->contexts.resize(count);
        p->threads.reserve(count);
        auto entry = reinterpret_cast<void* (*)(void*)>(workerFunction);
        for (unsigned int i = 0; i < count; ++i) {
            p->contexts[i] = {argument, p, i, 0};
            p->threads.emplace_back(entry, &p->contexts[i]);
        }
    } catch (...) {
        // Join any workers already started if thread creation/allocation fails.
        threadpoolDestroy(p);
        return 0;
    }
    std::unique_lock<std::mutex> lock(p->mutex);
    p->complete.wait(lock, [&] { return p->ready == count; });
    p->initialized = true;
    return 1;
}
