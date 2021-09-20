#ifndef STRINGEXCEPTION_H
#define STRINGEXCEPTION_H

#include <iostream>
#include <exception>
#include <stdexcept>
#include <sstream>

class StringException : public std::exception {
public:
    StringException(std::string msg) : message(msg) {}

    virtual const char* what() const throw() {
        return message.c_str();
    }
private:
    std::string message;
};

#endif // STRINGEXCEPTION_H
