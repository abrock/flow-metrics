#include "basename.h"

std::string basename(std::string name) {
    char * _name = (char*)name.c_str();
    char * _bname = basename(_name);
    return std::string(_bname);
}
