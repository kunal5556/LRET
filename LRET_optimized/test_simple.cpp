#include <iostream>
#include "gates_and_noise.h"
#include "utils.h"

using namespace qlret;

int main() {
    try {
        std::cout << "Starting simple test..." << std::endl;
        
        size_t num_qubits = 2;
        std::cout << "Creating zero state for " << num_qubits << " qubits..." << std::endl;
        
        MatrixXcd L = create_zero_state(num_qubits);
        std::cout << "Zero state created. Dimensions: " << L.rows() << "x" << L.cols() << std::endl;
        
        std::cout << "Test passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
