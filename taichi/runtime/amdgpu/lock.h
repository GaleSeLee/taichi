#pragma once
#include <chrono>
#include <mutex>
#include <condition_variable>

class AMDGPUlock {
    public:
        AMDGPUlock(std::mutex &mu) {

        }
    private:
        std::mutex mu_;
}