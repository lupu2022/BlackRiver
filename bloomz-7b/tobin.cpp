#include <msgpack.hpp>
#include <fstream>
#include <iostream>

template<typename T>
void load_data(const char* weights_file, std::vector<T> &allPerform) {
    std::ifstream t(weights_file, std::ios::binary);

    t.seekg(0, std::ios::end);
    size_t totalSize = t.tellg();
    t.seekg(0, std::ios::beg);

    char* memblock;
    memblock = new char [totalSize];
    t.read(memblock, totalSize);
    t.close();

    try {
        auto oh = msgpack::unpack((const char*)memblock, totalSize);
        allPerform = oh.get().as<std::vector<T>>();
        delete memblock;
    }
    catch (...) {
        std::cout << "Unpack weight error!" << std::endl;
        assert(false);
    }
}

int main( int argc, const char* argv[]) {
    std::cout << "Converting from " << argv[1] << " to " << argv[2] << std::endl;

    std::vector<int> data;
    load_data(argv[1], data);

    std::ofstream os(argv[2], std::ios::binary);
    os.write((const char *)data.data(), sizeof(int) * data.size());
}
