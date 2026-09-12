// No models or GPU needed: exercise kick/wait barriers, repeated generations,
// timeouts, idle destruction, and reuse of the Windows C++ worker pool.
#include "windowsWorkerPool.h"
#include <atomic>
#include <cstdio>

struct TestWork {
    int values[4]{};
    int increment = 0;
    std::atomic<bool> blocked{false};
};

static void* worker(void* arg)
{
    auto* ctx = static_cast<threadContext*>(arg);
    auto* work = static_cast<TestWork*>(ctx->argumentToPass);
    threadpoolWorkerInitialWait(ctx);
    while (threadpoolWorkerLoopCondition(ctx)) {
        while (work->blocked.load()) std::this_thread::yield();
        work->values[ctx->threadID] += work->increment;
        threadpoolWorkerLoopEnd(ctx);
    }
    return nullptr;
}

int main()
{
    workerPool pool;
    TestWork work;
    auto fail = [&](const char* message) {
        std::fprintf(stderr, "%s\n", message);
        work.blocked = false;
        threadpoolDestroy(&pool);
        return 1;
    };
    if (threadpoolCreate(&pool, 0, reinterpret_cast<void*>(&worker), &work))
        return fail("A zero-worker pool must be rejected");
    int expected = 0;
    for (int cycle = 0; cycle < 2; ++cycle) {
        if (!threadpoolCreate(&pool, 4, reinterpret_cast<void*>(&worker), &work))
            return fail("Could not create/recreate pool");
        for (int round = 1; round <= 2000; ++round) {
            work.increment = round;
            if (!threadpoolMainThreadPrepareWorkForWorkers(&pool) ||
                !threadpoolMainThreadKickWorkers(&pool) ||
                !threadpoolMainThreadWaitForKickedWorkersToFinishTimeoutSeconds(&pool, 5))
                return fail("Batch did not complete");
            expected += round;
            for (int value : work.values)
                if (value != expected) return fail("Lost or duplicated worker generation");
        }
        if (!threadpoolDestroy(&pool)) return fail("Idle pool destruction failed");
    }
    if (!threadpoolCreate(&pool, 4, reinterpret_cast<void*>(&worker), &work))
        return fail("Could not create timeout-test pool");
    work.increment = 1;
    work.blocked = true;
    if (!threadpoolMainThreadKickWorkers(&pool)) return fail("Timeout kick failed");
    if (threadpoolMainThreadWaitForKickedWorkersToFinishTimeoutSeconds(&pool, 1))
        return fail("Blocked work did not time out");
    work.blocked = false;
    if (!threadpoolMainThreadWaitForKickedWorkersToFinishTimeoutSeconds(&pool, 5))
        return fail("Timed-out work could not be harvested");
    threadpoolDestroy(&pool);
    std::puts("Windows worker pool: 4000 generations, timeout and reuse passed");
    return 0;
}
