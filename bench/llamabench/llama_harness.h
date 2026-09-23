#pragma once

#include <functional>
#include <string>
#include <vector>

using LlamaLog = std::function<void(const std::string&)>;

// Runs the engine-comparison protocol on a GGUF model with llama.cpp (CPU only). Prints RESULT lines.
bool llama_harness_run(const std::string& model_path, const std::string& label, const std::string& prompt,
                       const std::vector<int>& threads, const LlamaLog& log);
