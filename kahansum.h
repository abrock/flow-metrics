#ifndef KAHANSUM_H
#define KAHANSUM_H

class KahanSum {
    typedef double T;
    T c = 0;
    T t = 0;
    T y = 0;
    T sum = 0;

public:
    void push(T val) {
        y = val - c;
        t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    T getSum() {
        return sum;
    }
};

#endif // KAHANSUM_H
