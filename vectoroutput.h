#ifndef VECTOROUTPUT_H
#define VECTOROUTPUT_H

#include <vector>
#include <iostream>

template<class C>
std::ostream& operator << (std::ostream& out, const std::vector<C>& cont) {
    for (auto it: cont) {
        out << it << " ";
    }
    return out;
}

#endif // VECTOROUTPUT_H
