#include <chrono>
#include <cstdlib>
#include <iostream>
#include <torch/torch.h>

using namespace torch;

int main() {
  putenv("PYTORCH_CUDA_DSA_STACKTRACING=1");
  putenv("PYTORCH_USE_CUDA_DSA=1");

  torch::manual_seed(1);

  torch::Device device(torch::kCUDA);
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available!" << std::endl;
    device = torch::Device(torch::kCUDA);
  } else {
    std::cout << "CUDA is not available!" << std::endl;
  }

  std::cout << "Warmup" << std::endl;

  auto x = torch::rand({1024, 1024}, device);
  auto y = torch::multinomial(x, 2, true);
  auto result = y.cpu();

  std::cout << "Measuring..." << std::endl;
  auto start = std::chrono::steady_clock::now();
  try {
    for (int i = 0; i < 1000; ++i) {
      auto y = torch::multinomial(x, 1024, true);
      auto result = y.cpu();
    }
  } catch (const std::exception &e) {
    std::cout << e.what() << std::endl;
  }
  const auto duration = std::chrono::steady_clock::now() - start;
  std::cout << "Done " << std::chrono::duration_cast<std::chrono::microseconds>(duration).count() << std::endl;
}
