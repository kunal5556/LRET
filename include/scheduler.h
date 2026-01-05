#pragma once

/**
 * @file scheduler.h
 * @brief Phase 8.3+ Advanced Scheduling Strategies for Distributed Simulations
 *
 * Provides:
 * - FIFO baseline scheduler
 * - Adaptive load-aware scheduler
 * - Priority queue for gate operations
 */

#include "types.h"
#include <queue>
#include <vector>
#include <functional>
#include <memory>

namespace qlret {

//==============================================================================
// Scheduling Strategies
//==============================================================================

enum class SchedulePolicy {
    FIFO,      ///< First-in, first-out (baseline)
    ADAPTIVE,  ///< Load-aware dynamic reordering
    PRIORITY   ///< Priority-based (lower = higher priority)
};

struct ScheduledOp {
    size_t op_index = 0;         ///< Index in original sequence
    int priority = 0;            ///< Priority (used by PRIORITY policy)
    double estimated_cost = 1.0; ///< Estimated compute cost (used by ADAPTIVE)
};

/**
 * @brief Abstract scheduler interface.
 */
class Scheduler {
public:
    virtual ~Scheduler() = default;

    /**
     * @brief Submit an operation for scheduling.
     */
    virtual void submit(ScheduledOp op) = 0;

    /**
     * @brief Get next operation to execute.
     * @return Next operation, or nullopt if queue is empty
     */
    virtual std::optional<ScheduledOp> next() = 0;

    /**
     * @brief Check if scheduler has pending operations.
     */
    virtual bool empty() const = 0;

    /**
     * @brief Number of pending operations.
     */
    virtual size_t size() const = 0;
};

/**
 * @brief FIFO scheduler (baseline).
 */
class FIFOScheduler : public Scheduler {
public:
    void submit(ScheduledOp op) override { queue_.push(op); }
    std::optional<ScheduledOp> next() override {
        if (queue_.empty()) return std::nullopt;
        auto op = queue_.front();
        queue_.pop();
        return op;
    }
    bool empty() const override { return queue_.empty(); }
    size_t size() const override { return queue_.size(); }

private:
    std::queue<ScheduledOp> queue_;
};

/**
 * @brief Adaptive scheduler: prefers lower estimated_cost operations.
 */
class AdaptiveScheduler : public Scheduler {
public:
    void submit(ScheduledOp op) override { heap_.push(op); }
    std::optional<ScheduledOp> next() override {
        if (heap_.empty()) return std::nullopt;
        auto op = heap_.top();
        heap_.pop();
        return op;
    }
    bool empty() const override { return heap_.empty(); }
    size_t size() const override { return heap_.size(); }

private:
    struct CostCmp {
        bool operator()(const ScheduledOp& a, const ScheduledOp& b) const {
            return a.estimated_cost > b.estimated_cost;  // min-heap
        }
    };
    std::priority_queue<ScheduledOp, std::vector<ScheduledOp>, CostCmp> heap_;
};

/**
 * @brief Priority scheduler: lower priority value = higher priority.
 */
class PriorityScheduler : public Scheduler {
public:
    void submit(ScheduledOp op) override { heap_.push(op); }
    std::optional<ScheduledOp> next() override {
        if (heap_.empty()) return std::nullopt;
        auto op = heap_.top();
        heap_.pop();
        return op;
    }
    bool empty() const override { return heap_.empty(); }
    size_t size() const override { return heap_.size(); }

private:
    struct PriorityCmp {
        bool operator()(const ScheduledOp& a, const ScheduledOp& b) const {
            return a.priority > b.priority;  // min-heap by priority
        }
    };
    std::priority_queue<ScheduledOp, std::vector<ScheduledOp>, PriorityCmp> heap_;
};

/**
 * @brief Factory to create scheduler by policy.
 */
inline std::unique_ptr<Scheduler> create_scheduler(SchedulePolicy policy) {
    switch (policy) {
        case SchedulePolicy::FIFO: return std::make_unique<FIFOScheduler>();
        case SchedulePolicy::ADAPTIVE: return std::make_unique<AdaptiveScheduler>();
        case SchedulePolicy::PRIORITY: return std::make_unique<PriorityScheduler>();
        default: return std::make_unique<FIFOScheduler>();
    }
}

}  // namespace qlret
