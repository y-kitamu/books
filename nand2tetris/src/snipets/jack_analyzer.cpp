#include <iostream>
#include <memory>
#include <string>

#include "argparse/argparse.hpp"
#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"
#include "fmt/core.h"
#include "glog/logging.h"
#include "gperftools/profiler.h"

#include "lexical_analizer/compilation_engine.hpp"
#include "lexical_analizer/jack_tokenizer.hpp"

namespace fs = boost::filesystem;


bool endwith(const std::string& target, const std::string& suffix) {
    return target.substr(target.size() - suffix.size()) == suffix;
}

std::string replace_string_with(const std::string& str, const std::string& query,
                                const std::string& target) {
    auto pos = str.find(query);
    std::string res = str;
    res.replace(pos, pos + query.length(), target);
    return res;
}


bool check_same(const fs::path& path0, const fs::path& path1) {
    std::ifstream ifs0(path0.generic_string()), ifs1(path1.generic_string());
    if (!ifs0.is_open() || !ifs1.is_open()) {
        fmt::print("Failed to open file : {} or {}\n", path0.generic_string(), path1.generic_string());
        return false;
    }

    bool flag = true;
    while (true) {
        std::string s0, s1;
        ifs0 >> s0;
        ifs1 >> s1;

        flag &= s0 == s1;
        if (!flag) {
            fmt::print("Different : {} vs {}\n", s0, s1);
        }
        if (ifs0.eof() || ifs1.eof()) {
            break;
        }
    }
    auto print_res = [](std::ifstream& ifs) {
        std::string s;
        while (!ifs.eof()) {
            ifs >> s;
            fmt::print("Different : {}\n", s);
        }
    };
    if (!ifs0.eof()) {
        print_res(ifs0);
    }
    if (!ifs1.eof()) {
        print_res(ifs1);
    }
    flag &= ifs0.eof() && ifs1.eof();
    return flag;
}


void run_tokenizer(fs::path input) {
    auto get_output_path = [](fs::path input) {
        return input.parent_path() /
               replace_string_with(input.filename().generic_string(), ".jack", "_gen.xml");
    };

    auto get_gt_path = [](fs::path input) {
        return input.parent_path() /
               replace_string_with(input.filename().generic_string(), ".jack", "T.xml");
    };

    if (fs::is_directory(input)) {
        BOOST_FOREACH (const fs::path& p,
                       std::make_pair(fs::directory_iterator(input), fs::directory_iterator())) {
            if (!fs::is_directory(p) && p.extension() == ".jack") {
                fmt::print("Start processing file : {}\n", p.generic_string());
                auto tokenizer = la::JackTokenizer(p.generic_string());
                auto output = get_output_path(p);
                tokenizer.run_tokenize(output.generic_string());
                fmt::print("Finish processing file : {}\n", p.generic_string());

                auto gt = get_gt_path(p);
                if (check_same(gt, output)) {
                    fmt::print("Validation Success! : {}\n", output.generic_string());
                } else {
                    fmt::print("Validation Failed! : {}\n", output.generic_string());
                }
            }
        }
    } else {
        auto tokenizer = la::JackTokenizer(input.generic_string());
        auto output = get_output_path(input);
        tokenizer.run_tokenize(output.generic_string());
        fmt::print("Finish processing file : {}\n", input.generic_string());

        auto gt = get_gt_path(input);
        if (check_same(gt, output)) {
            fmt::print("Validation Success! : {}\n", output.generic_string());
        } else {
            fmt::print("Validation Failed! : {}\n", output.generic_string());
        }
    }
}


void run_compile(fs::path input) {
    fmt::print("start compile {}\n", input.generic_string());
    auto get_output_path = [](fs::path input) {
        return input.parent_path() /
               replace_string_with(input.filename().generic_string(), ".jack", "_compile.xml");
    };
    auto get_gt_path = [](fs::path input) {
        return input.parent_path() /
               replace_string_with(input.filename().generic_string(), ".jack", ".xml");
    };

    if (fs::is_directory(input)) {
        BOOST_FOREACH (const fs::path& p,
                       std::make_pair(fs::directory_iterator(input), fs::directory_iterator())) {
            if (!fs::is_directory(p) && p.extension() == ".jack") {
                run_compile(p);
            }
        }
    } else {
        auto output = get_output_path(input);
        auto gt = get_gt_path(input);
        auto compiler = la::CompilationEngine(input.generic_string(), output.generic_string());
        compiler.run_compile(true);
        fmt::print("Finish Processing file : {}\n", input.generic_string());
        if (check_same(gt, output)) {
            fmt::print("Validation Success! : {}\n", output.generic_string());
        } else {
            fmt::print("Validation Failed! : {}\n", output.generic_string());
        }
    }
}


int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();

    auto current_dir = fs::absolute(__FILE__).parent_path();
    auto project_root = current_dir.parent_path().parent_path();
    auto default_file = project_root / "projects" / "10" / "ArrayTest" / "Main.jack";

    fmt::print("file name : {}\n", __FILE__);
    fmt::print("current dir : {}\n", current_dir.generic_string());
    fmt::print("project root : {}\n", project_root.generic_string());
    fmt::print("default file : {}\n", default_file.generic_string());

    argparse::ArgumentParser program("Jack Analyzer");
    program.add_argument("-s")
        .default_value(default_file.generic_string())
        .help("Input source file or directory");

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        fmt::print("{}\n", err.what());
        std::cout << program;
        std::exit(0);
    }

    fs::path input = fs::path(program.get<std::string>("-s"));

    // run_tokenizer(input);
    run_compile(input);
}
