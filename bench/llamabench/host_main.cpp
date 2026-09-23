// PC check of the llama.cpp harness: llamabench model.gguf prompt.txt [threads...]
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "llama_harness.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: llamabench model.gguf prompt.txt [threads...]\n");
        return 2;
    }
    std::ifstream f(argv[2], std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    std::vector<int> threads;
    for (int i = 3; i < argc; ++i) threads.push_back(std::atoi(argv[i]));
    if (threads.empty()) threads = {2, 4};
    return llama_harness_run(argv[1], argv[1], ss.str(), threads,
                             [](const std::string& s) { std::printf("%s\n", s.c_str()); })
               ? 0
               : 1;
}
