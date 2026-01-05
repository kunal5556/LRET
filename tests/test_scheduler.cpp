#include "scheduler.h"
#include <cassert>
#include <iostream>

using namespace qlret;

int main() {
    try {
        // Test FIFO scheduler
        {
            auto sched = create_scheduler(SchedulePolicy::FIFO);
            sched->submit({0, 0, 1.0});
            sched->submit({1, 0, 2.0});
            sched->submit({2, 0, 0.5});

            assert(sched->size() == 3);
            auto op = sched->next();
            assert(op.has_value() && op->op_index == 0);
            op = sched->next();
            assert(op.has_value() && op->op_index == 1);
            op = sched->next();
            assert(op.has_value() && op->op_index == 2);
            assert(sched->empty());
            std::cout << "[PASS] FIFO scheduler\n";
        }

        // Test Adaptive scheduler (min estimated_cost first)
        {
            auto sched = create_scheduler(SchedulePolicy::ADAPTIVE);
            sched->submit({0, 0, 3.0});
            sched->submit({1, 0, 1.0});
            sched->submit({2, 0, 2.0});

            auto op = sched->next();
            assert(op.has_value() && op->op_index == 1);  // lowest cost
            op = sched->next();
            assert(op.has_value() && op->op_index == 2);
            op = sched->next();
            assert(op.has_value() && op->op_index == 0);
            assert(sched->empty());
            std::cout << "[PASS] Adaptive scheduler\n";
        }

        // Test Priority scheduler (min priority first)
        {
            auto sched = create_scheduler(SchedulePolicy::PRIORITY);
            sched->submit({0, 5, 1.0});
            sched->submit({1, 1, 1.0});
            sched->submit({2, 3, 1.0});

            auto op = sched->next();
            assert(op.has_value() && op->op_index == 1);  // priority 1
            op = sched->next();
            assert(op.has_value() && op->op_index == 2);  // priority 3
            op = sched->next();
            assert(op.has_value() && op->op_index == 0);  // priority 5
            assert(sched->empty());
            std::cout << "[PASS] Priority scheduler\n";
        }

        std::cout << "Scheduler tests passed\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << "\n";
        return 1;
    }
}
